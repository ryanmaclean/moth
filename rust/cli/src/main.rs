//! One-shot agent runner.
//!
//! `agent "your prompt"` — sends the prompt to Anthropic through the
//! harness, with vshell + audit on the shell side. Prints the model's
//! text and exits. Reads `ANTHROPIC_API_KEY` from env.

use std::process::ExitCode;
use std::sync::Arc;
use std::sync::mpsc::sync_channel;

use actor::spawn;
use harness::{
    AnthropicModel, AuditedShell, HarnessState, Instance, Sandbox, Session, SessionMsg,
};

const DEFAULT_MODEL: &str = "claude-haiku-4-5";

fn main() -> ExitCode {
    let mut args = std::env::args();
    let prog = args.next().unwrap_or_else(|| "agent".to_string());
    let prompt = args.collect::<Vec<_>>().join(" ");
    if prompt.is_empty() {
        eprintln!("usage: {prog} <prompt>");
        return ExitCode::from(2);
    }

    let api_key = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(k) => k,
        Err(_) => {
            eprintln!("ANTHROPIC_API_KEY required");
            return ExitCode::from(2);
        }
    };
    let model_name =
        std::env::var("MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

    let model = Arc::new(AnthropicModel::new(api_key, model_name));
    let sandbox: Box<dyn Sandbox> = Box::new(AuditedShell::new(vshell::VShell::new()));
    let inst = spawn(Instance::new("cli", sandbox));
    let state = Arc::new(HarnessState::new("default", model, inst.addr.clone()));
    let sess = spawn(Session::new("default", state));

    let (tx, rx) = sync_channel(1);
    if let Err(e) = sess.addr.send(SessionMsg::Prompt {
        text: prompt,
        structured_output_tag: None,
        reply: tx,
    }) {
        eprintln!("send failed: {e}");
        return ExitCode::from(1);
    }

    let exit = match rx.recv() {
        Ok(Ok(pr)) => {
            print!("{}", pr.text);
            if !pr.text.ends_with('\n') {
                println!();
            }
            if pr.completed {
                eprintln!("[completion signal fired]");
            }
            ExitCode::SUCCESS
        }
        Ok(Err(e)) => {
            eprintln!("session error: {e:?}");
            ExitCode::from(1)
        }
        Err(_) => {
            eprintln!("session dropped reply channel");
            ExitCode::from(1)
        }
    };

    let _ = sess.join();
    let _ = inst.join();
    exit
}
