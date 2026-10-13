# Video script — Claim Triage Assistant (target ≈ 4:15, hard limit 5:00)

Pace: about 130 words per minute. Text in **[brackets]** is what to show on screen; everything else is what
you say. Before recording, follow `DOC/Demo_Instructions.md` (warm up the models, and run case 03 once in the
UI so it appears under *Previous runs*).

---

### 0:00 – 0:20 · The problem
**[Show the application home page.]**

Hi, I'm Rodrigo, and this is my solution for the Bain take-home.
An insurance company receives property-damage claims with a description, photos and documents, and adjusters
need to triage them quickly and consistently. So I built a claim triage assistant.

### 0:20 – 0:40 · The solution and a note about the data
**[Point to the header that shows the models and "coordinator: llm".]**

It is a multi-agent system with a ReAct loop, a vector knowledge base of past claims, and a light web UI.
Everything runs locally with Ollama, so there are no API keys.
One note: the assessment folder I downloaded only had empty directories. So I created a synthetic knowledge
base of forty claims, my own demo cases and my own tests, and all of it is labelled as synthetic.

### 0:40 – 0:55 · Start the demo claim
**[Open "Load a demo case…" and pick "Ceiling leak around a light fitting". Scroll over the description and the attached photo and PDF. Click "Run triage".]**

In this claim, a pipe burst upstairs and water dripped out of the bedroom light fitting. There is a photo and a
repair estimate, uploaded like any user file. Let's run it.

### 0:55 – 1:30 · Architecture (while the run is in progress)
**[Show the "ReAct execution trace" section while the first steps appear.]**

While it runs: there are five agents, each with one job.
The intake analyst reads the description, the photo with a vision model, and the document.
The knowledge researcher searches the vector database.
The evidence auditor runs deterministic consistency checks.
The triage assessor computes the priority and writes the summary.
And the coordinator runs the ReAct loop.

### 1:30 – 2:05 · The ReAct loop
**[Point to one trace row: the action, the "decided by: llm" chip, the Reason line and the observation.]**

Each row is one Reason–Act–Observe step recorded by the backend. The coordinator only sees actions that are
valid in the current state; for example, it cannot assess before searching. The model picks one action with a
short reason, the tool runs, and the result goes back into the state. If the model gives an invalid answer, a
deterministic fallback decides, and a step limit stops any loop.

### 2:05 – 2:35 · Vector search of similar claims
**[When completed, scroll to "Similar historical claims". Highlight HC-2024-0040 and its similarity.]**

These are the similar historical claims. Each one is embedded with nomic-embed-text, and we search by cosine
similarity with a minimum threshold. The query combines the description and what the vision model saw.
For each match we show the score, the outcome and why it is similar. The closest one is also a pipe leak
through a ceiling near a light.

### 2:35 – 3:15 · The triage report
**[Scroll up to the P1 badge and the "Human review recommended" box. Then show "Priority factors", "Facts and their sources" and "Evidence consistency checks".]**

The priority is P1. It does not come from the language model: it comes from a transparent rubric, and every
point has a source, like the safety hazard declared by the claimant.
Every fact is labelled as declared, observed in a photo, from a document, or historical.
The auditor confirms that the claimed amount matches the estimate, and it flags that the photo looks synthetic,
which is correct, because our demo images are drawings.

### 3:15 – 3:35 · A suspicious claim
**[Open "Previous runs" and click CLM-DEMO-003. Show the review box and the checks.]**

In this second claim, the photo shows no damage, the invoice is dated before the incident, and a close
precedent was flagged for fraud. The system does not reject it. It sends it to a human, with the reasons.

### 3:35 – 4:05 · Validation and limitations
**[Show the bottom of the report: uncertainties, the confidence note and the disclaimer.]**

The report is validated with Pydantic, and the model's text is rejected if it decides coverage or cites claims
that were not retrieved. I wrote 52 automated tests, and all four demo cases pass their evaluation checks.
The main limitations are the small local models, which sometimes make mistakes, the speed on my laptop, and the
synthetic data.

### 4:05 – 4:15 · Close
**[Back to the top of the report.]**

So this is a working, traceable assistant that supports the adjuster and never replaces their decision.
Thank you for watching.

---

The triage run takes about 30–70 s on the development laptop, which the architecture section covers. If it is
still running at 1:30, keep talking about the trace rows that are already visible.
