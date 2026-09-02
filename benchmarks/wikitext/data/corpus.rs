use crate::wikitext::candle_msg;
use candle_core::Result;
use serde::Deserialize;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;

const PAGE: usize = 100;
const ROWS_URL: &str = "https://datasets-server.huggingface.co/rows?dataset=Salesforce/wikitext&config=wikitext-2-raw-v1&split=test";

#[derive(Deserialize)]
struct Page {
    rows: Vec<RowWrap>,
    num_rows_total: usize,
}

#[derive(Deserialize)]
struct RowWrap {
    row: TextRow,
}

#[derive(Deserialize)]
struct TextRow {
    text: String,
}

fn cache_path() -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push("quantize-wikitext-2-raw-v1-test.txt");
    path
}

pub fn wikitext2_test() -> Result<String> {
    let path = cache_path();
    if let Ok(text) = fs::read_to_string(&path) {
        if !text.is_empty() {
            return Ok(text);
        }
    }
    let text = download()?;
    if let Ok(mut file) = fs::File::create(&path) {
        let _ = file.write_all(text.as_bytes());
    }
    Ok(text)
}

fn download() -> Result<String> {
    let first = fetch_page(0)?;
    let total = first.num_rows_total;
    let mut text = String::new();
    append_rows(&mut text, &first);
    let mut offset = first.rows.len();
    while offset < total {
        let page = fetch_page(offset)?;
        if page.rows.is_empty() {
            break;
        }
        offset += page.rows.len();
        append_rows(&mut text, &page);
    }
    Ok(text)
}

fn fetch_page(offset: usize) -> Result<Page> {
    let url = format!("{ROWS_URL}&offset={offset}&length={PAGE}");
    let body = ureq::get(&url)
        .set("User-Agent", "quantize-wikitext/0.2")
        .call()
        .map_err(candle_msg)?
        .into_string()
        .map_err(candle_msg)?;
    serde_json::from_str(&body).map_err(candle_msg)
}

fn append_rows(text: &mut String, page: &Page) {
    for wrap in &page.rows {
        text.push_str(&wrap.row.text);
        if !wrap.row.text.ends_with('\n') {
            text.push('\n');
        }
    }
}

pub fn tokenize(tokenizer: &Tokenizer) -> Result<Vec<u32>> {
    let text = wikitext2_test()?;
    let encoding = tokenizer.encode(text, false).map_err(candle_msg)?;
    Ok(encoding.get_ids().to_vec())
}
