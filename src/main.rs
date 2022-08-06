use anyhow::{anyhow, Result as AnyhowResult};
use itertools::Itertools;
use tch::{Device, Tensor};
use tokenizers::tokenizer::Tokenizer;
use tokenizers::utils::padding::{pad_encodings, PaddingParams};
use tokenizers::PaddingStrategy;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;


const DEFAULT_SIZE: usize = 512;

fn main() -> AnyhowResult<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    tracing::info!("Running the tokenizer...");
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).map_err(|e| anyhow!(e))?;
    let encoded_tokens = tokenizer
        .encode("failed to create the tokenizer", true)
        .map_err(|e| anyhow!(e))?;

    let mut padding = PaddingParams::default();
    padding.strategy = PaddingStrategy::Fixed(DEFAULT_SIZE);

    tracing::info!("Padding encoded vectors to 512");
    let mut encoded_tokens_vecs = vec![encoded_tokens];
    pad_encodings(&mut encoded_tokens_vecs[..], &padding).map_err(|e| anyhow!(e))?;
    let encoded_tokens = encoded_tokens_vecs.remove(0);
    tracing::info!("Tokens: {:?}", encoded_tokens);

    let device = Device::cuda_if_available();
    let model = tch::jit::CModule::load_on_device("models/classification.pt", device)?;

    tracing::info!("IDs tokens");
    let ids = encoded_tokens
        .get_ids()
        .into_iter()
        .map(|x| *x as i32)
        .collect_vec();
    let ids = Tensor::of_slice(&ids[..]);
    let ids = ids.reshape(&[DEFAULT_SIZE as i64, 1]);
    tracing::info!("Word IDs:");
    ids.print();

    let mask = encoded_tokens
        .get_attention_mask()
        .into_iter()
        .map(|x| *x as i32)
        .collect_vec();
    let mask = Tensor::of_slice(&mask[..]);
    let mask = mask.reshape(&[DEFAULT_SIZE as i64, 1]);
    tracing::info!("Words mask:");
    mask.print();

    tracing::info!("Inference");
    let result = model.forward_ts(&[ids, mask])?;

    println!("Result classification:");
    result.print();

    Ok(())
}
