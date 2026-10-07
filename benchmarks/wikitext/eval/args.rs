use crate::wikitext::candle_msg;

/// `--max-tokens N` scores only the first N tokens. Any other argument is an
/// error, so a typo can't start the full run, which takes hours.
pub fn max_tokens() -> candle_core::Result<Option<usize>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [] => Ok(None),
        [flag, count] if flag == "--max-tokens" => count
            .parse()
            .map(Some)
            .map_err(|_| candle_msg(format!("--max-tokens needs a whole number, not `{count}`"))),
        _ => Err(candle_msg(format!(
            "expected `--max-tokens N` or nothing, not `{}`",
            args.join(" ")
        ))),
    }
}
