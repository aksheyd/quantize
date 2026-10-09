//! Split [`matmul_into`](Quantized::matmul_into) across every core, with the
//! `rayon` feature.
//!
//! The work splits into shares that the cores multiply at the same time, each
//! into its own pieces of `out`. A share costs more than its multiplications,
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
/// split costs a busy program less than a tenth more. On an idle 8-core x86
/// machine, two shares this size take 0.57 as long as one. An 18-core M5 Max
/// gains less the more threads its pool has, 0.6 as long with 2 threads but
/// 0.86 to 0.92 with all 18, though a split is never slower there either.
const VALUES_PER_ROW_SHARE: usize = 1 << 19;

/// Decoding a row takes about as long as multiplying it by this many vectors
/// for the [`whole_groups_of_32`] layouts, so a share of rows takes about as
/// long as multiplying its values by its vectors plus this many.
const DECODE_IN_VECTORS: usize = 4;

/// The fewest vectors in each part of a batch's vectors.
///
/// Each part decodes the whole matrix, which takes about as long as
/// multiplying it by 1 to 4 vectors for the [`whole_groups_of_32`] layouts,
/// and by up to 23 for the rest. When other threads already keep every core
/// busy, the shares take turns with them, which costs more than that. With 8
/// threads each multiplying their own batch on an 8-core x86 machine, parts
/// this size keep every call within about a seventh of its time without the
/// feature, where parts half this size cost up to a fifth.
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

    // A share multiplies some of the matrix's rows by some of the vectors:
    // the vectors split into parts, and so do the rows, and each share is one
    // part of each.
    //
    // A part of the rows decodes only those rows, so the rows split to give
    // every core a share, as long as each share takes about as long as one
    // vector through `VALUES_PER_ROW_SHARE` values.
    let fewest_work_per_share = VALUES_PER_ROW_SHARE * (1 + DECODE_IN_VECTORS);
    let row_parts = |vector_parts: usize| {
        let vectors_per_share = batch.div_ceil(vector_parts);
        let work = quantized
            .len()
            .saturating_mul(vectors_per_share + DECODE_IN_VECTORS);
        (cores / vector_parts)
            .min(work / fewest_work_per_share)
            .max(1)
    };
    // A part of the vectors decodes the whole matrix for itself, so each part
    // needs a full share of vectors. Of the ways to split them, take the one
    // that gives the most shares, and of those, the most parts of the
    // vectors: a smaller part's inputs stay in cache while its rows pass.
    // `max_by_key` returns the last of equal maxima, which has the most parts.
    let most_vector_parts = (batch / fewest_vectors_per_share(quantized, columns)).clamp(1, cores);
    let vector_parts = (1..=most_vector_parts)
        .max_by_key(|&vector_parts| vector_parts * row_parts(vector_parts))
        .unwrap_or(1);
    let vectors_per_share = batch.div_ceil(vector_parts);
    let rows_per_share = rows.div_ceil(row_parts(vector_parts));

    // A call too small for two shares stays on this thread.
    if vectors_per_share == batch && rows_per_share == rows {
        matmul_into(
            quantized,
            0,
            inputs,
            columns,
            &mut out.chunks_mut(rows).collect::<Vec<_>>(),
        );
        return;
    }
    // Cut each vector's results where the parts of the rows meet, and give
    // each piece to the share that multiplies those rows by that vector.
    let mut shares = Vec::new();
    let parts_of_the_batch = inputs
        .chunks(vectors_per_share * columns)
        .zip(out.chunks_mut(vectors_per_share * rows));
    for (part_inputs, part_out) in parts_of_the_batch {
        let mut pieces: Vec<Vec<&mut [f32]>> = (0..rows.div_ceil(rows_per_share))
            .map(|_| Vec::new())
            .collect();
        for vector_out in part_out.chunks_mut(rows) {
            for (row_part, piece) in vector_out.chunks_mut(rows_per_share).enumerate() {
                pieces[row_part].push(piece);
            }
        }
        for (row_part, share_out) in pieces.into_iter().enumerate() {
            shares.push((part_inputs, row_part * rows_per_share, share_out));
        }
    }
    shares
        .into_par_iter()
        .for_each(|(share_inputs, first_row, mut share_out)| {
            matmul_into(quantized, first_row, share_inputs, columns, &mut share_out)
        });
}
