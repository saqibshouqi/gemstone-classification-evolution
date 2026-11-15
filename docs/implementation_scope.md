# How the request and attached documents are used

## User request

The requested work is to create a durable project structure for the gemstone
classification coursework and complete **Person 1: Data and Scoreboard**. That
is the implementation scope of this change.

## Implementation design document

The design document supplies the research protocol and team contracts: the
2015/2022/2025 comparison, Chow's fixed 68-class data split, shared prediction
format, duplicate audit, metrics, and package boundaries. These are project
requirements used to shape the code; they are not separate messages directing
the assistant. The G1/G2/G3 directories are prepared as integration boundaries,
while those members' model implementations remain for their respective owners.

## Coursework brief

The coursework brief describes the assessed work: Task 5 demonstrates a
historical and a recent implementation, the report is limited to 20 A4 pages,
and the group presents a viva while students are assessed individually. The
brief also allows full GenAI use and asks for a record of prompts and use. Those
assessment requirements are documented here but do not expand this request into
implementing all three model arms or writing the report. The current AI-use
record is in `docs/ai_usage_log.md`.
