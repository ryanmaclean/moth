// AST node types. A "word" is a sequence of segments that get joined into
// one string at expansion time; this is what lets `foo$(bar)baz` work.

#[derive(Debug, Clone)]
pub enum Seg {
    Literal(String),
    // $VAR / ${VAR}
    Var(String),
    // $(...) — parsed as a full script
    CmdSub(Box<Node>),
}

#[derive(Debug, Clone)]
pub struct Word {
    pub segs: Vec<Seg>,
    // double-quoted segments keep their text intact (no later splitting),
    // single-quoted are pure Literal. Unquoted Literals are fine too;
    // we don't do field splitting after expansion anyway.
    pub quoted: bool,
}

#[derive(Debug, Clone)]
pub enum Redirect {
    StdoutTrunc(Word),
    StdoutAppend(Word),
    Stdin(Word),
    StderrTrunc(Word),
    StderrAppend(Word),
    StderrToStdout,
}

#[derive(Debug, Clone)]
pub struct Assign {
    pub name: String,
    pub value: Word,
}

#[derive(Debug, Clone)]
pub struct SimpleCommand {
    pub assigns: Vec<Assign>,
    pub words: Vec<Word>,
    pub redirects: Vec<Redirect>,
}

#[derive(Debug, Clone, Copy)]
pub enum SeqOp {
    Semi,
    And,
    Or,
}

#[derive(Debug, Clone)]
pub enum Node {
    Simple(SimpleCommand),
    Pipeline(Vec<SimpleCommand>),
    Sequence(Vec<(Node, SeqOp)>, Box<Node>),
}
