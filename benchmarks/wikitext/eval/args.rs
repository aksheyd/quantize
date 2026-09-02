pub fn max_tokens() -> Option<usize> {
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--max-tokens" {
            return args.next().and_then(|value| value.parse().ok());
        }
    }
    None
}
