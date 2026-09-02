pub struct Row {
    pub method: &'static str,
    pub perplexity: f64,
    pub tokens_per_second: f64,
}

pub fn print(rows: &[Row]) {
    println!();
    println!("{:<16}{:>12}{:>12}", "method", "perplexity", "tok/s");
    println!("{:-<16}{:->12}{:->12}", "", "", "");
    for row in rows {
        println!(
            "{:<16}{:>12.3}{:>12.2}",
            row.method, row.perplexity, row.tokens_per_second
        );
    }
}
