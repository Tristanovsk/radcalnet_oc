/*
 * Slides of the gas absorption-scattering coupling analysis (style of the GRS case-study decks).
 *
 * Usage:
 *   python extract_figures.py ../../../notebook/research/radcalnet_oc_gaseous_coupling_analysis.ipynb figures
 *   node make_slides.js out.pptx figures
 *
 * Requires pptxgenjs (npm install pptxgenjs, or NODE_PATH pointing to a node_modules folder with it).
 * The figures are those of the executed analysis notebook, in their order of appearance (figure_1.png,
 * ...): figure_1 transmittance of the windows, figure_2 eta_atm vs absorption strength, figure_4 band
 * error vs solar angle. The numbers of the slides are those of the notebook run of 2026-11-12 (set v2).
 */
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const OUT = process.argv[2];
const FIG = process.argv[3] || "figures";
if (!OUT) {
  console.error("usage: node make_slides.js out.pptx [figure_dir]");
  process.exit(2);
}
const FIGURES = { spectrum: "figure_1.png", saturation: "figure_2.png", geometry: "figure_4.png" };
for (const f of Object.values(FIGURES)) {
  if (!fs.existsSync(path.join(FIG, f))) {
    console.error(`missing ${path.join(FIG, f)}: run extract_figures.py on the executed notebook first`);
    process.exit(1);
  }
}

// template palette (grstbx_bolmon.pptx)
const H = {
  title: "123B4A", text: "1B2B34", grey: "5E6E77", dark: "434343", green: "2E7D32", tint: "EAF3EA",
  amber: "C77D05", moss: "3F7D3C", brown: "8D6E63", note: "FDF3E1", grid: "D9E2DC", white: "FFFFFF",
};
const THEME = {
  name: "GRS case study",
  headFontFace: "Comfortaa",
  bodyFontFace: "Arial",
  colors: {
    dk1: H.text, lt1: H.white, dk2: H.title, lt2: H.tint,
    accent1: H.green, accent2: H.amber, accent3: H.grey, accent4: H.moss, accent5: H.brown, accent6: H.dark,
    hlink: H.green, folHlink: H.moss,
  },
};
const FOOTER = "RadCalNet-OC · SOS-ABS coupled simulations (CKD 5 cm⁻¹) · gas_coupling v2, November 2026";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Gaseous absorption and scattering coupling in RadCalNet-OC";
pres.author = "Tristan Harmel";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "COVER",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.25, w: 12.1, h: 1.75, fontSize: 48, bold: true, color: C.accent6, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.1, w: 12.1, h: 1.2, fontSize: 22, fontFace: "Comfortaa", color: C.accent6, valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.6, fontSize: 32, bold: true, color: C.text2, valign: "middle", align: "left" }, text: "" } },
    { text: { text: FOOTER, options: { x: 0.6, y: 7.0, w: 9.0, h: 0.3, fontSize: 10, fontFace: "Calibri", color: C.accent3 } } },
  ],
  slideNumber: { x: 11.73, y: 7.0, w: 1.0, h: 0.3, fontSize: 10, fontFace: "Calibri", color: C.accent3, align: "right" },
});

const T = (s, t) => s.addText(t, { placeholder: "title" });
const box = (s, text, o) => s.addText(text, { isTextBox: true, margin: 0, valign: "top", fontSize: 16, color: C.text1, ...o });
// bullets with a bold green lead-in, as in the template
const lead = (items, o = {}) => items.flatMap(([b, t], i) => {
  const last = i === items.length - 1;
  const runs = [{ text: b, options: { bold: true, color: o.leadColor || C.accent1, bullet: o.bullet !== false } }];
  runs.push({ text: t, options: { breakLine: !last } });
  return runs;
});
const outlined = (s, x, y, w, h, col, name) =>
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.background1 }, line: { color: col, width: 3 }, objectName: name });
const disc = (s, x, y, d, col, label, name) => {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: col }, line: { type: "none" }, objectName: name });
  if (label) box(s, label, { x, y, w: d, h: d, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle", fontFace: "Calibri", objectName: name + "_n" });
};
const tile = (s, x, y, w, big, lab, name, f = 0.5) => {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h: 1.25, fill: { color: C.background2 }, line: { type: "none" }, objectName: name });
  box(s, big, { x: x + 0.15, y: y + 0.1, w: w * f, h: 1.05, fontSize: 26, bold: true, fontFace: "Cambria", color: C.accent1, valign: "middle", objectName: name + "_v" });
  box(s, lab, { x: x + 0.25 + w * f, y: y + 0.1, w: w * (1 - f) - 0.35, h: 1.05, fontSize: 14, fontFace: "Calibri", color: C.accent3, valign: "middle", objectName: name + "_l" });
};
const caption = (s, t, o) => box(s, t, { fontSize: 14, fontFace: "Calibri", color: C.accent3, ...o });
const img = (s, file, x, y, w, pxw, pxh, alt) => s.addImage({ path: path.join(FIG, file), x, y, w, h: w * pxh / pxw, altText: alt });
const TH = (t) => ({ text: t, options: { bold: true, color: H.white, fill: { color: H.title } } });
const tableOpts = (o) => ({ fontSize: 14, fontFace: "Calibri", color: H.text, border: { type: "solid", pt: 0.5, color: H.grid }, ...o });
const band = (rows) => rows.map((r, i) => r.map((c) => (typeof c === "object" ? { text: c.text, options: { fill: { color: i % 2 ? H.white : H.tint }, ...c.options } } : { text: c, options: { fill: { color: i % 2 ? H.white : H.tint } } })));
const chartBase = {
  catAxisLabelColor: H.text, valAxisLabelColor: H.text, catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt",
  valGridLine: { color: H.grid, size: 0.5 }, catGridLine: { style: "none" }, legendFontSize: 12, legendPos: "t",
  dataLabelFontSize: 9, dataLabelColor: H.text, titleColor: H.text, titleFontSize: 13,
};
const BANDS = ["O2-B 687", "H2O 722", "OLCI Oa13", "OLCI Oa14", "OLCI Oa15", "H2O 822", "OLCI Oa19", "OLCI Oa20", "MSI B9"];

// ===== 1. cover =====
pres.addSection({ title: "Overview" });
let s = pres.addSlide({ masterName: "COVER", sectionTitle: "Overview" });
s.addText([{ text: "Gas absorption and scattering", options: { breakLine: true } }, { text: "coupled in RadCalNet-OC" }], { placeholder: "title" });
s.addText("Effective air masses for the O2 and H2O bands of 684–990 nm, from 27 500 coupled SOS-ABS simulations, and their implementation in the package", { placeholder: "body" });
img(s, FIGURES.spectrum, 0.85, 3.55, 11.6, 1337, 419, "Direct transmittance of the simulated absorption windows for three water vapour columns");
s.addNotes("Companion deck of notebook/research/radcalnet_oc_gaseous_coupling_analysis.ipynb. Image: CKD direct transmittance at air mass 2 in the simulated windows, water vapour 0.5, 2 and 4 g cm-2.");

// ===== 2. key messages =====
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Overview" });
T(s, "Key messages");
const km = [
  [C.accent1, "A large bias in the bands", "The decoupled model underestimates the band TOA reflectance by 4 % (O2-B) to 50–60 % (940 nm)."],
  [C.text2, "Corrected to 1–6 %", "Effective air masses and a single-scattering blend, fitted on coupled runs and cross-validated: Oa20 57 → 6 %, Oa13 30 → 3 %."],
  [C.accent4, "Applied at 5 cm⁻¹", "The correction is computed at the resolution of the fit, so it transfers to REPTRAN and line-by-line O2."],
  [C.accent5, "Tables matter too", "The gas tables differ by up to 18 %: use gas_coupling='effective_air_mass' with o2_model='lbl'."],
];
km.forEach(([col, h, t], i) => {
  const x = i % 2 ? 6.75 : 0.6, y = i < 2 ? 1.5 : 4.2;
  outlined(s, x, y, 5.95, 2.45, col, `km${i}`);
  disc(s, x + 0.3, y + 0.32, 0.55, col, String(i + 1), `kmd${i}`);
  box(s, h, { x: x + 1.05, y: y + 0.3, w: 4.65, h: 0.6, fontSize: 24, bold: true, fontFace: "Calibri", valign: "middle", objectName: `kmh${i}` });
  box(s, t, { x: x + 1.05, y: y + 1.0, w: 4.65, h: 1.3, fontSize: 19, fontFace: "Calibri", objectName: `kmt${i}` });
});

// ===== 3. problem =====
pres.addSection({ title: "Physics and simulations" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Physics and simulations" });
T(s, "Decoupled model: exact for the direct beam only");
s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.4, w: 12.1, h: 1.2, fill: { color: C.background2 }, line: { type: "none" }, objectName: "eqbg" });
box(s, [
  { text: "R_toa = Tg(m) · R_atm + Tg(m) · t_d · t_u · π R_rs        E_d = Tg(m0) · T_d · E0", options: { breakLine: true, fontFace: "Cambria", fontSize: 20, color: C.text2 } },
  { text: "m0 = 1/μ0, m = 1/μ0 + 1/μv: air mass of the direct Sun-surface-sensor path", options: { fontSize: 14, fontFace: "Calibri", color: C.accent3 } },
], { x: 0.85, y: 1.55, w: 11.6, h: 0.95, objectName: "eq" });
const eff = [
  [C.accent1, "Path reflectance", ["Photons scattered above the absorbers shortcut the path", "Less absorption, mostly for H2O (scale height ≈ 2 km)", "η_atm ≈ 0.4–0.7"]],
  [C.accent2, "Water-leaving, Ed", ["Skylight reaches the surface on slanted paths", "More absorption than the direct beam", "η_w, η_Ed ≈ 1.02–1.05"]],
  [C.text2, "Saturation", ["CKD bands: sums of exponentials", "−ln Tg not proportional to m", "Shortcut shrinks as lines saturate"]],
];
eff.forEach(([col, h, items], i) => {
  const x = 0.6 + i * 4.1;
  outlined(s, x, 2.95, 3.9, 3.75, col, `eff${i}`);
  disc(s, x + 0.25, 3.15, 0.4, col, null, `effd${i}`);
  box(s, h, { x: x + 0.8, y: 3.05, w: 3.0, h: 0.6, fontSize: 22, bold: true, color: C.accent6, valign: "middle", objectName: `effh${i}` });
  box(s, items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < items.length - 1, bold: j === 2, color: j === 2 ? col : C.accent6 } })),
      { x: x + 0.25, y: 3.85, w: 3.45, h: 2.7, fontSize: 17, paraSpaceAfter: 8, objectName: `efft${i}` });
});
s.addNotes("η = effective air mass / geometric air mass; η = 1 is the decoupled model.");

// ===== 4. simulations =====
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Physics and simulations" });
T(s, "27 500 coupled runs over five absorption regions");
tile(s, 0.6, 1.4, 2.85, "23 220", ["v2 runs (fit)", "+ 4 272 v1 (test)"].join("\n"), "t0");
tile(s, 3.7, 1.4, 2.85, "0.5–4", ["H2O column", "g cm⁻²"].join("\n"), "t1");
tile(s, 6.8, 1.4, 2.85, "20–70°", ["solar zenith", "angle"].join("\n"), "t2");
tile(s, 9.9, 1.4, 2.85, "0.05–0.4", ["AOT(550)", "maritime"].join("\n"), "t3", 0.58);
s.addTable(band([
  [TH("set"), TH("sza (°)"), TH("AOT(550)"), TH("H2O (g cm⁻²)"), TH("albedo ρ")],
  ["v1 (test)", "30, 60", "0.05, 0.2", "2", "0, 0.01, 0.03, 0.1"],
  ["v2 (fit)", "20, 40, 55, 70", "0.05, 0.15, 0.4", "0.5, 2, 4", "0, 0.03, 0.1"],
]).map((r, i) => (i === 0 ? [TH("set"), TH("sza (°)"), TH("AOT(550)"), TH("H2O (g cm⁻²)"), TH("albedo ρ")] : r)),
  tableOpts({ x: 0.6, y: 2.95, w: 7.2, colW: [1.2, 1.5, 1.5, 1.3, 1.7], rowH: 0.4, objectName: "simtable" }));
caption(s, "US62, 1013 hPa, WMO maritime aerosols (2 km), Cox-Munk 5 m s⁻¹ + Lambertian ρ, CKD 5 cm⁻¹ fine mode. v1 shares no solar angle with v2.",
  { x: 0.6, y: 4.35, w: 7.2, h: 0.8, objectName: "simcap" });
box(s, lead([
  ["R(ρ) = R0 + ρA / (1 − Sρ)", " per geometry, with and without absorption"],
  ["T_atm = R0 ratio, T_w = A ratio", "; direct sunglint removed from R0, as in the LUTs"],
  ["η = m*(T) / m", " with the CKD direct table Tg(λ, U, m)"],
]), { x: 8.1, y: 2.95, w: 4.6, h: 3.6, fontSize: 16, paraSpaceAfter: 10, objectName: "method" });
caption(s, "Fine CKD mode: one radiative transfer run per k term, which couples absorption and scattering exactly.",
  { x: 0.6, y: 5.35, w: 7.2, h: 0.9, objectName: "ckdcap" });

// ===== 5. eta =====
pres.addSection({ title: "Coupling" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Coupling" });
T(s, "Path reflectance sees half the air mass");
const WIN = ["O2-B", "H2O 720", "O2-A head", "O2-A P", "H2O 820", "H2O 900", "H2O 940"];
s.addChart(pres.charts.BAR, [
  { name: "η atm (path reflectance)", labels: WIN, values: [0.711, 0.545, 0.476, 0.616, 0.551, 0.487, 0.414] },
  { name: "η w (water-leaving)", labels: WIN, values: [1.045, 1.050, 1.021, 1.035, 1.046, 1.037, 1.027] },
], { ...chartBase, x: 0.4, y: 1.35, w: 7.6, h: 5.4, barDir: "col", barGapWidthPct: 60, chartColors: [H.green, H.amber],
     showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1.2,
     showTitle: true, title: "median η = m*/m over v2 (η = 1: decoupled)", objectName: "etachart" });
box(s, lead([
  ["Half the air mass", " for the path reflectance: η_atm 0.41 (940 nm) to 0.71 (O2-B)"],
  ["Slightly more", " for the water-leaving term and Ed: 1.02–1.05"],
  ["Aerosols raise η_atm", ": 0.31 → 0.53 at 940 nm for AOT 0.05 → 0.4 (scattering near the surface)"],
  ["Water vapour lowers it", ": 0.50 → 0.35 for 0.5 → 4 g cm⁻² (saturation)"],
]), { x: 8.3, y: 1.45, w: 4.5, h: 5.3, fontSize: 16, paraSpaceAfter: 12, objectName: "etatext" });

// ===== 6. saturation =====
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Coupling" });
T(s, "The shortcut shrinks as lines saturate");
img(s, FIGURES.saturation, 0.6, 1.4, 12.1, 2100, 493, "eta_atm against absorption strength per window, coloured by aerosol load, with the single-scattering curve");
box(s, lead([
  ["Single scattering", ", absorber mixed like the scatterers: T_ss = (1 − Tg)/(−ln Tg), η = 1/2 for weak lines, less when saturated (black curve)"],
  ["More aerosols", " push η_atm above the curve; clean air in the H2O bands falls below it (absorber lower than the molecules)"],
  ["A linear η", " in m and AOT cannot follow this in the most saturated windows"],
]), { x: 0.6, y: 4.5, w: 12.1, h: 2.3, fontSize: 16, paraSpaceAfter: 8, objectName: "sattext" });

// ===== 7. forms =====
pres.addSection({ title: "Parameterization" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Parameterization" });
T(s, "Two forms, one coefficient per window and term");
[[C.accent1, "Effective air mass (eta)", "T = Tg(η m),  η = c", "all terms; robust to the gas table", 0.6],
 [C.accent2, "Single-scattering blend", "T = Tg(m)^(1−k) · T_ss^k,  k = c", "path reflectance, saturated windows", 6.75]].forEach(([col, h, f, t, x], i) => {
  outlined(s, x, 1.4, 5.95, 1.85, col, `form${i}`);
  disc(s, x + 0.3, 1.62, 0.4, col, null, `formd${i}`);
  box(s, h, { x: x + 0.85, y: 1.55, w: 4.9, h: 0.55, fontSize: 22, bold: true, color: C.accent6, valign: "middle", objectName: `formh${i}` });
  box(s, f, { x: x + 0.3, y: 2.2, w: 5.4, h: 0.45, fontSize: 19, fontFace: "Cambria", color: C.text2, objectName: `formf${i}` });
  caption(s, t, { x: x + 0.3, y: 2.7, w: 5.4, h: 0.4, objectName: `formc${i}` });
});
box(s, "c = c0 + c1 (m − 2) + c2 (τ550 − 0.1) + c3 ln(U / 2 g cm⁻²)", { x: 0.6, y: 3.5, w: 12.1, h: 0.5, fontSize: 20, fontFace: "Cambria", color: C.text2, objectName: "cform" });
caption(s, "Fits weighted for the absolute transmittance error, which drives band averages; direct glint removed.", { x: 0.6, y: 4.0, w: 8.4, h: 0.4, objectName: "wnote" });
const wins = [["O2-B", 684, 698, 0], ["H2O 720", 712, 736, 0], ["O2-A head", 757, 762.25, 1], ["O2-A P", 762.25, 772, 0],
              ["H2O 820", 808, 836, 0], ["H2O 900", 890, 919.5, 0], ["H2O 940", 919.5, 990, 1]];
const X0 = 0.9, X1 = 12.4, L0 = 680, L1 = 995, sx = (l) => X0 + (l - L0) / (L1 - L0) * (X1 - X0);
s.addShape(pres.shapes.LINE, { x: X0, y: 5.5, w: X1 - X0, h: 0, line: { color: H.grey, width: 1 }, objectName: "axis" });
wins.forEach(([n, a, b, bl], i) => {
  s.addShape(pres.shapes.RECTANGLE, { x: sx(a), y: 4.7, w: sx(b) - sx(a), h: 0.8, fill: { color: bl ? C.accent2 : C.accent1 }, line: { color: H.white, width: 0.75 }, objectName: `win${i}` });
  const lx = (sx(a) + sx(b)) / 2;
  box(s, n, { x: lx - 0.65, y: i % 2 ? 5.95 : 5.6, w: 1.3, h: 0.3, fontSize: 12, fontFace: "Calibri", align: "center", objectName: `winl${i}` });
});
[700, 750, 800, 850, 900, 950].forEach((l, i) => caption(s, `${l} nm`, { x: sx(l) - 0.4, y: 6.4, w: 0.8, h: 0.25, fontSize: 11, align: "center", objectName: `tick${i}` }));
box(s, [{ text: "■ ", options: { color: C.accent1 } }, { text: "eta    " }, { text: "■ ", options: { color: C.accent2 } }, { text: "blend (path reflectance)" }],
    { x: 9.0, y: 4.05, w: 3.7, h: 0.3, fontSize: 12, fontFace: "Calibri", align: "right", objectName: "legend" });

// ===== 8. choice =====
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Parameterization" });
T(s, "Forms chosen on cross-validated band errors");
const sc = [["O2-B", "1.96", "2.75", "eta"], ["H2O 720", "3.57", "4.49", "eta"], ["O2-A head", "12.22", "10.46", "blend"], ["O2-A P", "3.81", "4.31", "eta"],
            ["H2O 820", "3.94", "4.87", "eta"], ["H2O 900", "4.96", "6.92", "eta"], ["H2O 940", "31.89", "19.87", "blend"]];
s.addTable([[TH("window"), TH("score eta"), TH("score blend"), TH("adopted")],
  ...band(sc.map(([w, a, b, ch]) => [w, { text: a, options: { bold: ch === "eta" } }, { text: b, options: { bold: ch === "blend" } },
    { text: ch, options: { bold: true, color: ch === "blend" ? H.amber : H.green } }]))],
  tableOpts({ x: 0.6, y: 1.4, w: 6.4, colW: [1.9, 1.5, 1.6, 1.4], rowH: 0.45, objectName: "scoretable" }));
caption(s, "Score: mean of the median and 95th percentile of |band error| (%), over the bands of the window and over the independent test (v1), leave-one-sza-out, leave-one-H2O-out and the transfer to REPTRAN and line-by-line O2.",
  { x: 0.6, y: 5.2, w: 6.4, h: 1.4, objectName: "scorecap" });
box(s, lead([
  ["Split windows", ": the O2-A head and the 940 nm core saturate far more than the P branch and the 900 nm shoulder"],
  ["One form per band", " was a compromise: the blend helps Oa13 but hurts Oa14, helps Oa20 but hurts Oa19"],
  ["Not retained", ": a blend with an absorption-strength term (c4 τ)"],
  ["Reproducible", ": python coupling.py fit writes the same table"],
]), { x: 7.4, y: 1.45, w: 5.3, h: 5.2, fontSize: 16, paraSpaceAfter: 12, objectName: "whytext" });

// ===== 9. band errors =====
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
T(s, "Band errors fall to 1–6 % in median");
s.addChart(pres.charts.BAR, [
  { name: "decoupled", labels: BANDS, values: [4.0, 11.2, 29.6, 14.6, 2.0, 8.9, 11.8, 56.8, 52.1] },
  { name: "first version (eta v1)", labels: BANDS, values: [0.7, 2.6, 8.0, 3.2, 1.7, 2.2, 3.4, 25.4, 21.4] },
  { name: "adopted (v2)", labels: BANDS, values: [0.9, 1.2, 3.2, 1.8, 1.1, 1.2, 1.7, 6.1, 5.5] },
], { ...chartBase, x: 0.4, y: 1.35, w: 8.6, h: 5.45, barDir: "col", barGapWidthPct: 50, chartColors: [H.amber, "9AA7AE", H.green],
     showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 65,
     showTitle: true, title: "median |error| of the band TOA reflectance (%), v2, clear water ρ = 0.005", objectName: "bandchart" });
box(s, lead([
  ["Independent test (v1)", ": Oa20 6.9 %, Oa13 3.0 %, others 0.9–2.0 %"],
  ["95th percentile", ": 2.5–11 %; 27–28 % at 940 nm"],
  ["Turbid water", " (ρ = 0.03): smaller errors, larger surface share"],
  ["Ed and water-leaving", ": below 1 % (decoupled 0.2–1.6 %)"],
]), { x: 9.3, y: 1.45, w: 3.5, h: 5.3, fontSize: 16, paraSpaceAfter: 12, objectName: "bandside" });

// ===== 10. geometry =====
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
T(s, "The blend holds where eta drifts");
img(s, FIGURES.geometry, 0.6, 1.3, 12.1, 1648, 576, "Band error against solar zenith angle for Oa13, 822 nm and Oa20, per model and water vapour column");
box(s, lead([
  ["Oa13 and Oa20", ": eta v2 drifts with sza and H2O (−12 % at 70°, −55 % at U = 4 in Oa20); the blends stay within about ±8 %"],
  ["822 nm", ": the v2 corrections stay within ±1 %; the decoupled bias grows from −3 % (U = 0.5) to −17 % (U = 4)"],
]), { x: 0.6, y: 5.65, w: 12.1, h: 1.2, fontSize: 16, paraSpaceAfter: 6, objectName: "geotext" });

// ===== 11. transfer =====
pres.addSection({ title: "Implementation" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Implementation" });
T(s, "Apply the correction at 5 cm⁻¹, not per channel");
s.addChart(pres.charts.BAR, [
  { name: "decoupled", labels: BANDS, values: [4.5, 12.5, 33.1, 16.6, 2.4, 10.4, 14.1, 62.6, 57.9] },
  { name: "package, lbl O2", labels: BANDS, values: [0.7, 1.6, 4.8, 5.2, 0.7, 2.4, 2.5, 8.6, 5.9] },
  { name: "package, REPTRAN", labels: BANDS, values: [1.9, 1.6, 4.7, 2.5, 2.8, 2.4, 2.5, 8.6, 5.9] },
], { ...chartBase, x: 0.4, y: 1.35, w: 8.0, h: 5.45, barDir: "col", barGapWidthPct: 50, chartColors: [H.amber, H.green, H.title],
     showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 70,
     showTitle: true, title: "path reflectance: |C_pkg / C_sim − 1| (%), median over v2", objectName: "transchart" });
tile(s, 8.7, 1.45, 4.0, "> 1000 %", "940 nm, blend applied per REPTRAN channel (T ≈ 1e-30)", "pc");
box(s, lead([
  ["GasCoupling.correction", ": smooth Tg to 5 cm⁻¹, correct there, apply per channel keeping the interval means"],
  ["Physical", ": absorptance scaled when absorption decreases, transmittance when it increases (0 ≤ T ≤ 1)"],
]), { x: 8.7, y: 2.95, w: 4.0, h: 3.8, fontSize: 16, paraSpaceAfter: 12, objectName: "corrtext" });

// ===== 12. tables =====
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Implementation" });
T(s, "Gas tables differ by up to 18 %");
s.addChart(pres.charts.BAR, [
  { name: "REPTRAN", labels: BANDS, values: [-14.1, -6.9, -16.2, -18.4, -9.1, -7.8, -8.1, -12.8, -7.6] },
  { name: "line-by-line O2", labels: BANDS, values: [0.5, -6.9, 16.1, 2.0, -1.3, -7.8, -8.1, -12.8, -7.6] },
], { ...chartBase, x: 0.4, y: 1.35, w: 6.8, h: 5.45, barDir: "col", barGapWidthPct: 50, chartColors: [H.title, H.green],
     showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", valAxisMinVal: -25, valAxisMaxVal: 20,
     showTitle: true, title: "direct-beam band Tg, package vs CKD (%), m = 2, U = 2", objectName: "tabchart" });
box(s, "End to end with lbl O2: band path reflectance vs coupled simulation, signed median (%)", { x: 7.5, y: 1.4, w: 5.2, h: 0.55, fontSize: 14, bold: true, fontFace: "Calibri", color: C.text2, objectName: "e2eh" });
s.addTable([[TH("band"), TH("decoupled"), TH("package")], ...band([
  ["O2-B 687", "−3.9", "+0.4"], ["OLCI Oa13", "−20.5", "+14.8"], ["OLCI Oa14", "−11.9", "−0.2"], ["OLCI Oa15", "−3.5", "−0.4"],
  ["H2O 722 / 822", "−19.6 / −19.2", "−5.8 / −5.7"], ["OLCI Oa19", "−22.3", "−6.1"], ["OLCI Oa20", "−68.3", "−7.2"], ["MSI B9", "−61.7", "−4.3"],
])], tableOpts({ x: 7.5, y: 2.0, w: 5.2, colW: [1.75, 1.8, 1.65], rowH: 0.38, objectName: "e2etable" }));
s.addShape(pres.shapes.RECTANGLE, { x: 7.5, y: 5.65, w: 5.2, h: 1.1, fill: { color: H.note }, line: { type: "none" }, objectName: "notebg" });
box(s, [{ text: "H2O tables also differ", options: { bold: true, color: C.accent2 } }, { text: " (2–14 %, growing with U): a line-by-line H2O reference is needed." }],
    { x: 7.7, y: 5.75, w: 4.8, h: 0.95, fontSize: 15, objectName: "note" });

// ===== 13. using it =====
pres.addSection({ title: "Conclusions" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusions" });
T(s, "Using the correction in RadCalNet-OC");
const cards = [
  [C.accent1, "What it does", ["Corrects Ratm, Lw and Ed in 7 windows, 684–990 nm", "Band errors 1–6 % (median)", "Works with REPTRAN and lbl O2"]],
  [C.accent2, "Limits", ["940 nm: 95th percentile ≈ 28 %", "Maritime aerosols, US62, 1013 hPa", "Gaps between windows uncorrected"]],
  [C.text2, "Next steps", ["Other aerosols, H2O profiles", "Line-by-line H2O reference", "Decide on switching the default"]],
];
cards.forEach(([col, h, items], i) => {
  const x = 0.6 + i * 4.1;
  outlined(s, x, 1.45, 3.9, 3.05, col, `uc${i}`);
  disc(s, x + 0.25, 1.65, 0.4, col, null, `ucd${i}`);
  box(s, h, { x: x + 0.8, y: 1.55, w: 3.0, h: 0.6, fontSize: 22, bold: true, color: C.accent6, valign: "middle", objectName: `uch${i}` });
  box(s, items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < items.length - 1 } })),
      { x: x + 0.25, y: 2.35, w: 3.45, h: 2.0, fontSize: 16, color: C.accent6, paraSpaceAfter: 6, objectName: `uct${i}` });
});
box(s, "In short", { x: 0.6, y: 4.75, w: 12.1, h: 0.5, fontSize: 25, bold: true, color: C.accent6, objectName: "inshort" });
box(s, [
  { text: "Use gas_coupling='effective_air_mass' with o2_model='lbl' for bands in the O2 and H2O windows", options: { bullet: true, breakLine: true } },
  { text: "The default stays 'decoupled' until other aerosol models and H2O profiles are checked", options: { bullet: true, breakLine: true } },
  { text: "Regenerate the table with scripts/sos_abs_coupling/coupling.py fit after new simulations", options: { bullet: true } },
], { x: 0.6, y: 5.3, w: 12.1, h: 1.55, fontSize: 18, color: C.accent6, paraSpaceAfter: 6, objectName: "inshorttext" });

// write the theme colors (pptxgenjs writes the default Office scheme) and make the chart axes portable
async function applyTheme(file, theme) {
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const slots = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink"];
  const scheme = `<a:clrScheme name="${theme.name}">` + slots.map((k) => `<a:${k}><a:srgbClr val="${theme.colors[k]}"/></a:${k}>`).join("") + "</a:clrScheme>";
  const part = "ppt/theme/theme1.xml";
  zip.file(part, (await zip.file(part).async("string")).replace(/<a:clrScheme\b[\s\S]*?<\/a:clrScheme>/, scheme)
    .replace(/(<a:(?:theme|fontScheme)\b[^>]*?\bname=")[^"]*"/g, (_, h) => `${h}${theme.name}"`));
  for (const name of Object.keys(zip.files)) {
    if (!name.startsWith("ppt/charts/") || !name.endsWith(".xml")) continue;
    const xml = await zip.file(name).async("string");
    if (xml.includes('<c:crossesAt val="autoZero"/>')) zip.file(name, xml.replaceAll('<c:crossesAt val="autoZero"/>', '<c:crosses val="autoZero"/>'));
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
})();
