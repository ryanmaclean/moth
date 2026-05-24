//! Parser. Tokens -> AST.

use crate::ShellError;
use crate::lex::{Tok, Word, WordPart, tokenize};

fn first_literal(argv: &[Word]) -> Option<String> {
    let first = argv.first()?;
    if first.len() == 1
        && let WordPart::Literal(s) = &first[0]
    {
        return Some(s.clone());
    }
    None
}

fn is_keyword(s: &str) -> bool {
    matches!(
        s,
        "if" | "then"
            | "else"
            | "elif"
            | "fi"
            | "case"
            | "esac"
            | "for"
            | "select"
            | "while"
            | "until"
            | "do"
            | "done"
            | "function"
            | "in"
    )
}

#[derive(Debug, Clone)]
pub(crate) struct SimpleCommand {
    /// `KEY=value` prefixes attached to this command (apply only for the run).
    pub assigns: Vec<(String, Word)>,
    pub argv: Vec<Word>,
    pub redirects: Vec<Redirect>,
}

#[derive(Debug, Clone)]
pub(crate) enum Redirect {
    StdoutTrunc(Word),
    StdoutAppend(Word),
    Stdin(Word),
    StderrTrunc(Word),
    StderrAppend(Word),
    Err2Out,
}

#[derive(Debug, Clone)]
pub(crate) struct Pipeline {
    pub stages: Vec<SimpleCommand>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SeqOp {
    /// `&&`
    AndIf,
    /// `||`
    OrIf,
}

#[derive(Debug, Clone)]
pub(crate) struct Sequence {
    /// `first op[0] step[0] op[1] step[1] ...`
    pub head: Pipeline,
    pub tail: Vec<(SeqOp, Pipeline)>,
}

#[derive(Debug, Clone)]
pub(crate) enum Item {
    /// Standalone `KEY=value` (no command) — sets shell variable.
    BareAssign(String, Word),
    Seq(Sequence),
}

pub(crate) fn parse(src: &str) -> Result<Vec<Item>, ShellError> {
    let toks = tokenize(src)?;
    let mut p = Parser { toks, pos: 0 };
    p.program()
}

struct Parser {
    toks: Vec<(Tok, usize)>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|(t, _)| t)
    }
    fn bump(&mut self) -> Option<(Tok, usize)> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn cur_pos(&self) -> usize {
        self.toks.get(self.pos).map(|(_, p)| *p).unwrap_or(0)
    }

    fn program(&mut self) -> Result<Vec<Item>, ShellError> {
        let mut items = Vec::new();
        while self.pos < self.toks.len() {
            // Skip empty separators.
            while matches!(self.peek(), Some(Tok::Semi)) {
                self.pos += 1;
            }
            if self.pos >= self.toks.len() {
                break;
            }
            // Standalone assignment? Only if a single `Assign` followed by Semi/EOF
            // AND there isn't a command word after it. Otherwise it's a prefix
            // of a SimpleCommand.
            if let Some(Tok::Assign(_, _)) = self.peek() {
                let save = self.pos;
                let mut assigns = Vec::new();
                while let Some(Tok::Assign(_, _)) = self.peek() {
                    if let Some((Tok::Assign(k, v), _)) = self.bump() {
                        assigns.push((k, v));
                    }
                }
                // What's next? If it's Word -> it's a command with env prefix; backtrack.
                match self.peek() {
                    None | Some(Tok::Semi) => {
                        for (k, v) in assigns {
                            items.push(Item::BareAssign(k, v));
                        }
                        continue;
                    }
                    _ => {
                        self.pos = save;
                    }
                }
            }
            let seq = self.sequence()?;
            items.push(Item::Seq(seq));
        }
        Ok(items)
    }

    fn sequence(&mut self) -> Result<Sequence, ShellError> {
        let head = self.pipeline()?;
        let mut tail = Vec::new();
        loop {
            let op = match self.peek() {
                Some(Tok::AndIf) => SeqOp::AndIf,
                Some(Tok::OrIf) => SeqOp::OrIf,
                _ => break,
            };
            self.pos += 1;
            // Skip newlines/semis after && or || (line continuation-ish).
            while matches!(self.peek(), Some(Tok::Semi)) {
                self.pos += 1;
            }
            let p = self.pipeline()?;
            tail.push((op, p));
        }
        // `;` and newline terminate the sequence; consumed at program level.
        Ok(Sequence { head, tail })
    }

    fn pipeline(&mut self) -> Result<Pipeline, ShellError> {
        let mut stages = vec![self.simple_command()?];
        while matches!(self.peek(), Some(Tok::Pipe)) {
            self.pos += 1;
            while matches!(self.peek(), Some(Tok::Semi)) {
                self.pos += 1;
            }
            stages.push(self.simple_command()?);
        }
        Ok(Pipeline { stages })
    }

    fn simple_command(&mut self) -> Result<SimpleCommand, ShellError> {
        let mut assigns = Vec::new();
        let mut argv = Vec::new();
        let mut redirects = Vec::new();
        // Leading assignments.
        while let Some(Tok::Assign(_, _)) = self.peek() {
            if let Some((Tok::Assign(k, v), _)) = self.bump() {
                assigns.push((k, v));
            }
        }
        loop {
            match self.peek() {
                Some(Tok::Word(_)) => {
                    if let Some((Tok::Word(w), _)) = self.bump() {
                        argv.push(w);
                    }
                }
                Some(Tok::Assign(_, _)) if argv.is_empty() => {
                    // Still in prefix territory; pull more assigns.
                    if let Some((Tok::Assign(k, v), _)) = self.bump() {
                        assigns.push((k, v));
                    }
                }
                Some(Tok::Assign(k, v)) => {
                    // `KEY=value` *after* the command name is just a word in POSIX.
                    let mut w: Word = vec![crate::lex::WordPart::Literal(format!("{k}="))];
                    w.extend(v.clone());
                    self.pos += 1;
                    argv.push(w);
                }
                Some(Tok::Great) => {
                    self.pos += 1;
                    redirects.push(Redirect::StdoutTrunc(self.expect_target()?));
                }
                Some(Tok::DGreat) => {
                    self.pos += 1;
                    redirects.push(Redirect::StdoutAppend(self.expect_target()?));
                }
                Some(Tok::Less) => {
                    self.pos += 1;
                    redirects.push(Redirect::Stdin(self.expect_target()?));
                }
                Some(Tok::Great2) => {
                    self.pos += 1;
                    redirects.push(Redirect::StderrTrunc(self.expect_target()?));
                }
                Some(Tok::DGreat2) => {
                    self.pos += 1;
                    redirects.push(Redirect::StderrAppend(self.expect_target()?));
                }
                Some(Tok::Err2Out) => {
                    self.pos += 1;
                    redirects.push(Redirect::Err2Out);
                }
                _ => break,
            }
        }
        if assigns.is_empty() && argv.is_empty() && redirects.is_empty() {
            return Err(ShellError::ParseError {
                msg: "expected a command".into(),
                pos: self.cur_pos(),
            });
        }
        if let Some(name) = first_literal(&argv)
            && is_keyword(&name)
        {
            return Err(ShellError::UnsupportedSyntax(format!(
                "shell keyword `{name}` (loops/conditionals/functions) not supported"
            )));
        }
        Ok(SimpleCommand { assigns, argv, redirects })
    }

    fn expect_target(&mut self) -> Result<Word, ShellError> {
        match self.bump() {
            Some((Tok::Word(w), _)) => Ok(w),
            other => Err(ShellError::ParseError {
                msg: format!("expected filename after redirect, got {other:?}"),
                pos: self.cur_pos(),
            }),
        }
    }
}
