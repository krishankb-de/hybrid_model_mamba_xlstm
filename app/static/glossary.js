// app/static/glossary.js — the plain explanations of the report card (CHAT_UI_PLAN.md P6-H): static data, nothing else.
//
// Each row is the words a report uses for a thing (the user's table gives the first, and the same thing said another way; the plural forms come
// after them, P6-H fix 1) and what they say in plain language. render.js marks these words in a report and explains each row once: the Terms
// list names a row by the words the report used where it first used them, in lower case. Its summary says the CheXbert labels in the same words
// (SUMMARY_LABELS). The explanations are the user's own table, word for word, so a change here is a change to what the page tells a reader.
// The tables are frozen, so that no script can change them at run time.

const row = (terms, explanation) => Object.freeze({ terms: Object.freeze(terms), explanation });

// The rows of the 14 CheXbert labels first, "no finding" to "support devices", then the words of a report that a reader meets most. The row
// "pleural thickening" marks that term in a report; it is not what the summary says of the label "Pleural Other" (SUMMARY_LABELS).
export const GLOSSARY = Object.freeze([
  row(['no finding'], 'no abnormality is reported'),
  row(['enlarged cardiomediastinum'], 'the heart and the central chest look wider than normal'),
  row(['cardiomegaly'], 'an enlarged heart'),
  row(['lung opacity', 'opacity', 'lung opacities', 'opacities'], 'an area of the lung that looks whiter, or denser, than normal'),
  row(['lung lesion', 'lesion', 'lung lesions', 'lesions'], 'a spot or mass in the lung'),
  row(['edema'], 'fluid in the lung tissue'),
  row(['consolidation', 'consolidations'], 'part of the lung filled with fluid or inflammation instead of air'),
  row(['pneumonia'], 'a lung infection'),
  row(['atelectasis'], 'part of a lung is partly collapsed or not fully inflated'),
  row(['pneumothorax', 'pneumothoraces'], 'air between the lung and the chest wall, which can make the lung collapse'),
  row(['pleural effusion', 'effusion', 'pleural effusions', 'effusions'], 'fluid between the lung and the chest wall'),
  row(['pleural thickening'], 'a thickened lining around the lung'),
  row(['fracture', 'fractures'], 'a broken bone, for example a rib'),
  row(['support devices', 'device', 'devices'], 'tubes, lines, wires or devices placed in the body'),
  row(['vascular congestion'], 'the blood vessels in the lungs look fuller than normal'),
  row(['hilar', 'hila'], 'around the roots of the lungs, where vessels and airways enter'),
  row(['basilar', 'base', 'bases'], 'the lower parts of the lungs'),
  row(['costophrenic angle', 'costophrenic angles'], 'the lower outer corner of the lung on the image'),
  row(['mediastinum', 'mediastinal'], 'the central chest between the lungs'),
  row(['interstitial'], 'in the tissue between the air sacs'),
  row(['nodule', 'nodules'], 'a small round spot'),
  row(['granuloma', 'granulomas'], 'a small healed spot, often from an old infection'),
  row(['hemidiaphragm', 'hemidiaphragms'], 'one half of the breathing muscle under the lungs'),
  row(['emphysema'], 'damaged air sacs that trap air'),
  row(['sternotomy wires'], 'wires that close the breastbone after chest surgery'),
  row(['endotracheal tube'], 'a breathing tube in the windpipe'),
  row(['nasogastric tube'], 'a tube through the nose into the stomach'),
  row(['picc', 'picc line', 'picc lines'], 'a thin tube in an arm vein that reaches near the heart'),
  row(['pacemaker'], 'a device that helps the heart keep a regular beat'),
  row(['scoliosis'], 'a sideways curve of the spine'),
  row(['osseous'], 'about the bones'),
  row(['focal'], 'in one small area'),
  row(['bilateral'], 'on both sides'),
  row(['acute'], 'new or sudden'),
  row(['chronic'], 'long-standing'),
]);

// What the summary says of a row: its plain phrase and then its first term in brackets, "an enlarged heart (cardiomegaly)".
const said = (term) => `${GLOSSARY.find((entry) => entry.terms[0] === term).explanation} (${term})`;

// The CheXbert label names of the summary, in the order it says them (the order of CHEXBERT_14, with Support Devices last), each with what the
// summary says of it. "No Finding" is not here: a report with it says no abnormal finding, in a sentence of its own (render.js summaryText).
// "Pleural Other" is not said as the row "pleural thickening": the label also covers scarring, calcification and other changes of the pleura,
// so it has words of its own, which no row of the glossary has.
export const SUMMARY_LABELS = Object.freeze([
  ['Enlarged Cardiomediastinum', said('enlarged cardiomediastinum')],
  ['Cardiomegaly', said('cardiomegaly')],
  ['Lung Opacity', said('lung opacity')],
  ['Lung Lesion', said('lung lesion')],
  ['Edema', said('edema')],
  ['Consolidation', said('consolidation')],
  ['Pneumonia', said('pneumonia')],
  ['Atelectasis', said('atelectasis')],
  ['Pneumothorax', said('pneumothorax')],
  ['Pleural Effusion', said('pleural effusion')],
  ['Pleural Other', 'a change in the lining around the lung other than fluid (other pleural finding)'],
  ['Fracture', said('fracture')],
  ['Support Devices', said('support devices')],
].map((pair) => Object.freeze(pair)));
