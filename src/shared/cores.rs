//! Split [`matmul_into`](Quantized::matmul_into) across every core, with the
//! `rayon` feature.
//!
//! The work splits into shares that the cores multiply at the same time, each
//! into its own slice of `out`. A share costs more than its multiplications,
//! so a call splits only into shares big enough to pay for themselves, and a
//! call too small for two stays on the thread that made it, as every call does
//! without the feature. That matters most when a program already multiplies
//! on several threads at once: its cores are busy, so a split can't make a call
//! faster, and whatever the split costs comes out of every call.

use rayon::prelude::*;

use crate::decode::{matmul_into, whole_groups_of_32};
use crate::scale::Scale;
use crate::tensor::Quantized;

/// The fewest values of the matrix in each share of one vector's rows.
///
/// Handing a share to another core costs about 7 µs of core time when every
/// core is already busy with calls of its own, and one vector takes about
/// 100 µs through 2^19 values of a `Q8_32` matrix, the fastest layout. So a
/// split costs a busy program less than a tenth more, while on an idle machine
/// two shares this size take a little over half as long as one.
const VALUES_PER_ROW_SHARE: usize = 1 << 19;

/// The fewest vectors in each share of a batch.
///
/// Each share decodes the whole matrix, which takes about as long as
/// multiplying it by 1 to 4 vectors for the [`whole_groups_of_32`] layouts,
/// and by up to 23 for the rest. When other threads already keep every core
/// busy, the shares take turns with them, which costs more than that. With 8
/// threads each multiplying their own batch on an 8-core x86 machine, shares
/// this size keep every call within about a seventh of its time without the
/// feature, where shares half this size cost up to a fifth.
fn fewest_vectors_per_share<S: Scale>(quantized: &Quantized<S>, columns: usize) -> usize {
    if whole_groups_of_32(quantized, columns) {
        64
    } else {
        184
    }
}

/// Multiply `inputs` by the `rows × columns` matrix into `out`, as
/// [`Quantized::matmul_into`] does, on every core when the call is big enough.
pub(crate) fn matmul_on_every_core<S: Scale>(
    quantized: &Quantized<S>,
    inputs: &[f32],
    rows: usize,
    columns: usize,
    out: &mut [f32],
) {
    let cores = rayon::current_num_threads();
    let batch = inputs.len() / columns;
    if batch == 1 {
        // One vector's results are one per row, and each depends on its row
        // alone, so the rows split into shares. Each share decodes only its
        // own rows, so the work doesn't grow.
        let shares = (quantized.len() / VALUES_PER_ROW_SHARE).min(cores);
        if shares > 1 {
            let rows_per_share = rows.div_ceil(shares);
            out.par_chunks_mut(rows_per_share)
                .enumerate()
                .for_each(|(share, out)| {
                    matmul_into(quantized, share * rows_per_share, inputs, columns, out)
                });
            return;
        }
    } else {
        // A vector's results depend on that vector alone, so a batch splits
        // into shares of vectors. Each share decodes the whole matrix, so a
        // large share still goes through it in groups that stay in cache, as
        // a large batch does.
        let shares = (batch / fewest_vectors_per_share(quantized, columns)).min(cores);
        if shares > 1 {
            let vectors_per_share = batch.div_ceil(shares);
            inputs
                .par_chunks(vectors_per_share * columns)
                .zip(out.par_chunks_mut(vectors_per_share * rows))
                .for_each(|(inputs, out)| matmul_into(quantized, 0, inputs, columns, out));
            return;
        }
    }
    matmul_into(quantized, 0, inputs, columns, out);
}
