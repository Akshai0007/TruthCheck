"""Web interface:  streamlit run app.py"""
import html
import json
import os

import streamlit as st

from truthcheck import LocalDocsRetriever, TruthCheck, WikipediaRetriever
from truthcheck.demo import DEMO_ANSWER, DEMO_DOCS, DEMO_QUESTION
from truthcheck.llm import KEY_ENV, find_api_key, make_llm, models_for
from truthcheck.retrieval import text_from_file

st.set_page_config(page_title="TruthCheck", page_icon="🔎", layout="wide")

# Colour is never the only signal: each verdict also has its own underline style and a text label.
st.markdown("""
<style>
.v-supported    {background: rgba(46,160,67,.22);  border-bottom: 2px solid #2ea043; padding: 0 2px;}
.v-contradicted {background: rgba(218,54,51,.22);  border-bottom: 2px wavy #da3633; padding: 0 2px;}
.v-unverified   {background: rgba(210,153,34,.25); border-bottom: 2px dotted #d29922; padding: 0 2px;}
.answer-box {line-height: 1.8; font-size: 1.05rem; max-width: 70ch;}
</style>""", unsafe_allow_html=True)

LABEL = {"supported": "Supported", "contradicted": "Contradicted", "unverified": "Couldn't verify"}
MARK = {"supported": "🟢", "contradicted": "🔴", "unverified": "🟡"}
SEVERITY = {"contradicted": 2, "unverified": 1, "supported": 0}


def highlighted(answer, claims):
    """Wrap each answer sentence in a span coloured by its worst verdict."""
    worst = {}
    for c in claims:
        if c.sentence and (c.sentence not in worst or SEVERITY[c.verdict] > SEVERITY[worst[c.sentence]]):
            worst[c.sentence] = c.verdict
    out = html.escape(answer)
    for sent, verdict in worst.items():
        esc = html.escape(sent)
        out = out.replace(esc, f'<span class="v-{verdict}">{esc}</span>', 1)
    return out


# ---------------- sidebar: settings ----------------
with st.sidebar:
    st.header("Settings")
    PROVIDERS = {"Groq": "groq", "OpenAI": "openai", "Gemini": "gemini"}
    options = list(PROVIDERS) + ["Demo (offline, sample only)"]
    preferred = os.getenv("TRUTHCHECK_PROVIDER", "").lower().strip()
    start = next((i for i, o in enumerate(options) if PROVIDERS.get(o, "demo") == preferred), None)
    if start is None:   # no valid default: first provider that has a key, else the offline demo
        start = next((i for i, o in enumerate(options) if PROVIDERS.get(o) and find_api_key(PROVIDERS[o])),
                     len(options) - 1)
    provider = st.selectbox("Language model", options, index=start)
    key = PROVIDERS.get(provider, "demo")
    if key == "demo":
        st.info("Demo mode is a script, not real checking. It only understands the built-in example.")
    elif not find_api_key(key):
        st.error(f"No API key found. Add {KEY_ENV[key][0]}=... to your .env file and restart the app.")
    else:
        st.caption("Models are picked automatically; if one is busy the next is tried: "
                   + " > ".join(models_for(key)))
    source = "Demo documents" if key == "demo" else st.radio("Where to look for evidence",
                                                             ["Wikipedia", "My own documents"])
    files = st.file_uploader("Upload .txt, .md or .pdf files", type=["txt", "md", "pdf"],
                             accept_multiple_files=True) if source == "My own documents" else []
    samples = st.slider("Re-ask the question (consistency check)", 0, 5, 0,
                        help="Asks the model the question again and flags claims it keeps changing. "
                             "Works in 'Ask a question' mode. Uses more calls.")
    policy = st.radio("Claims that can't be verified", ["Remove from answer", "Keep but mark"])

# ---------------- main ----------------
st.title("TruthCheck")
st.write("Paste an AI answer, or ask a question. Every factual claim is checked against evidence, "
         "and the answer is corrected where the evidence disagrees.")

mode = st.radio("What do you want to do?", ["Check an answer I paste", "Ask a question"], horizontal=True)
default = (DEMO_ANSWER if mode.startswith("Check") else DEMO_QUESTION) if key == "demo" else ""
text = st.text_area("Answer to check" if mode.startswith("Check") else "Your question",
                    value=default, height=140)
question = st.text_input("What was it answering? (optional)") if mode.startswith("Check") else None

if st.button("Check", type="primary", disabled=not text.strip()):
    try:
        llm = make_llm(key)
        if key == "demo":
            retriever = LocalDocsRetriever(DEMO_DOCS)
        elif source == "Wikipedia":
            retriever = WikipediaRetriever()
        elif files:
            retriever = LocalDocsRetriever({f.name: text_from_file(f.name, f.read()) for f in files})
        else:
            st.error("Upload at least one document, or switch to Wikipedia.")
            st.stop()
        tc = TruthCheck(llm, retriever, consistency_samples=samples,
                        unverified_policy="remove" if policy.startswith("Remove") else "flag")
        with st.spinner("Checking every claim..."):
            rep = tc.check(text, question or None) if mode.startswith("Check") else tc.ask_and_check(text)
    except Exception as e:
        st.error(f"Could not run the check: {e}")
        st.stop()
    st.session_state["report"] = rep
    st.session_state["models_used"] = getattr(getattr(llm, "inner", None), "usage", None) or {}

rep = st.session_state.get("report")
if rep:
    used = st.session_state.get("models_used")
    if used:
        st.caption("Answered by: " + ", ".join(f"{m} ({n} calls)" for m, n in used.items()))
    for n in rep.notes:
        st.warning(n)
    if rep.claims:
        c = rep.counts()
        a, b, d, e = st.columns(4)
        a.metric("Trust score", f"{rep.trust_score:.0%}", help="Share of claims backed by evidence")
        b.metric("Supported", c["supported"])
        d.metric("Contradicted", c["contradicted"])
        e.metric("Couldn't verify", c["unverified"])

        left, right = st.columns(2)
        with left:
            st.subheader("Original answer")
            st.markdown(f'<div class="answer-box">{highlighted(rep.answer, rep.claims)}</div>',
                        unsafe_allow_html=True)
        with right:
            st.subheader("Corrected answer")
            st.markdown(f'<div class="answer-box">{html.escape(rep.corrected_answer or "")}</div>',
                        unsafe_allow_html=True)

        st.subheader("Claim by claim")
        for cr in rep.claims:
            with st.expander(f"{MARK[cr.verdict]} {LABEL[cr.verdict]}: {cr.claim}"):
                st.write(cr.reasoning or "No explanation given.")
                if cr.corrected_claim:
                    st.write(f"**Evidence says:** {cr.corrected_claim}")
                if cr.consistency is not None:
                    st.write(f"Changed in {cr.consistency:.0%} of re-asked answers.")
                for n in cr.source_numbers:
                    s = next(x for x in rep.sources if x["n"] == n)
                    link = f" ([open]({s['url']}))" if s["url"] else ""
                    st.caption(f"[{n}] {s['source']}{link}: {s['text']}")
        if rep.skipped:
            st.caption("Not checked (opinion or advice): " + " | ".join(rep.skipped))
        st.download_button("Download full report (JSON)", json.dumps(rep.to_dict(), indent=2),
                           file_name="truthcheck_report.json", mime="application/json")
