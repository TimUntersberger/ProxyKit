const byId = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = byId<HTMLCanvasElement>("card-canvas");
const ctx = canvas.getContext("2d")!;
const outerBorderAssets = [
  { value: "super-rare", path: "/border_sr.png", image: new Image() },
  { value: "secret-rare", path: "/border_sec.png", image: new Image() },
];
let artwork: HTMLImageElement | null = null;
let artworkUrl: string | null = null;
let loadRevision = 0;

const cardFields = [
  "card-cost", "card-power", "card-counter", "card-name", "card-type",
  "card-frame-color", "card-effect-background", "card-effect", "card-number",
  "card-rarity", "card-block-number", "card-outer-border", "card-colored-border",
];
const borderDefaults: Record<string, string> = {
  "card-outer-border": "normal",
  "card-colored-border": "visible",
};

const attributes = [
  { label: "Strike", color: "#ad8528", kanji: "打" },
  { label: "Special", color: "#92367d", kanji: "特" },
  { label: "Slash", color: "#247fa4", kanji: "斬" },
  { label: "Wisdom", color: "#479e65", kanji: "知" },
];
const selectedAttributes = new Set([attributes[0]]);
const maxAttributes = 2;

const frameColors = {
  blue: {
    backing: ["#058dc3", "#0067ad"],
    rail: "#088ec6",
    capsule: { fill: "#006da9", edge: "#005783" },
    emblem: { fill: "#005880", highlight: "#009bd0", edge: "#004b7d", filledSection: 2 },
  },
  black: {
    backing: ["#353535", "#171717"],
    rail: "#1c1c1c",
    capsule: { fill: "#101010", edge: "#2b2b2b" },
    emblem: { fill: "#131313", highlight: "#272727", edge: "#464646", filledSection: 4, sectionColor: "#555555" },
  },
  green: {
    backing: ["#009c7b", "#006d55"],
    rail: "#008669",
    capsule: { fill: "#008669", edge: "#005b46" },
    emblem: { fill: "#00604a", highlight: "#13a087", edge: "#004c3b", filledSection: 1 },
  },
  red: {
    backing: ["#ed292d", "#ac1019"],
    rail: "#d71318",
    capsule: { fill: "#c01018", edge: "#920414" },
    emblem: { fill: "#86001c", highlight: "#ed4042", edge: "#ad1727", filledSection: 0 },
  },
  yellow: {
    backing: ["#fff174", "#d4b91e"],
    rail: "#ffe82c",
    text: "#101218",
    capsule: { fill: "#f3dd32", edge: "#aa9d2b" },
    emblem: { fill: "#aaa029", highlight: "#fff474", edge: "#d0ba32", filledSection: 5 },
  },
  purple: {
    backing: ["#a5579e", "#632b60"],
    rail: "#883c84",
    capsule: { fill: "#793474", edge: "#5f2a5a" },
    emblem: { fill: "#62365a", highlight: "#a5589d", edge: "#562d51", filledSection: 3 },
  },
};

const value = (id: string) => (byId<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>(id).value || "").trim();

function roundRect(x: number, y: number, w: number, h: number, r: number, fill: string | CanvasGradient, stroke?: string, lineWidth = 1) {
  ctx.beginPath(); ctx.roundRect(x, y, w, h, r); ctx.fillStyle = fill; ctx.fill();
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth; ctx.stroke(); }
}
function fitted(text: string, maxWidth: number, size: number, weight = 700): number {
  while (size > 18) { ctx.font = `${weight} ${size}px Arial, sans-serif`; if (ctx.measureText(text).width <= maxWidth) break; size -= 2; }
  return size;
}
function drawCover(image: HTMLImageElement, x: number, y: number, w: number, h: number) {
  const scale = Math.max(w / image.naturalWidth, h / image.naturalHeight);
  const sw = w / scale, sh = h / scale;
  ctx.drawImage(image, (image.naturalWidth - sw) / 2, (image.naturalHeight - sh) / 2, sw, sh, x, y, w, h);
}

function tracePowerShoulder() {
  ctx.lineTo(608, 43);
  ctx.lineTo(646, 106); ctx.quadraticCurveTo(651, 114, 664, 114);
}

function tracePowerHeaderEdge() {
  tracePowerShoulder();
  ctx.lineTo(947, 114); ctx.quadraticCurveTo(956, 114, 956, 123);
}

function traceFooterTop() {
  ctx.quadraticCurveTo(956, 1197, 947, 1197);
  ctx.lineTo(679, 1197); ctx.quadraticCurveTo(674, 1197, 671, 1193);
  ctx.lineTo(649, 1171); ctx.quadraticCurveTo(646, 1167, 641, 1167);
  ctx.lineTo(357, 1167); ctx.quadraticCurveTo(352, 1167, 349, 1171);
  ctx.lineTo(328, 1193); ctx.quadraticCurveTo(325, 1197, 320, 1197);
  ctx.lineTo(52, 1197); ctx.quadraticCurveTo(43, 1197, 43, 1188);
}

function traceFooterOutside() {
  ctx.lineTo(43, 1348); ctx.quadraticCurveTo(43, 1358, 53, 1358);
  ctx.lineTo(946, 1358); ctx.quadraticCurveTo(956, 1358, 956, 1348);
}

function traceArtworkOutline() {
  ctx.beginPath(); ctx.moveTo(43, 75); ctx.quadraticCurveTo(43, 43, 75, 43);
  tracePowerHeaderEdge();
  ctx.lineTo(956, 1188);
  traceFooterTop();
  ctx.closePath();
}
type EffectToken = { text: string; badge: boolean; bold: boolean; x: number; y: number; width: number; scale?: number };
function effectFont(size: number, badge: boolean, outlined: boolean, bold = false) {
  const bodyWeight = outlined ? 400 : 450;
  return `${badge ? 500 : bold ? 700 : bodyWeight} ${badge ? size * .85 : size}px Roboto, "Segoe UI", Arial, sans-serif`;
}

function effectLayout(source: string, x: number, top: number, width: number, size: number, outlined: boolean) {
  ctx.font = effectFont(size, false, outlined);
  const lineHeight = size * 1.4;
  const space = ctx.measureText(" ").width;
  const tokens: EffectToken[] = [];
  let y = top, cursor = 0;
  const lines = source.replace(/\r/g, "").split("\n");
  for (const line of lines) {
    if (!line.trim()) { y += lineHeight * .4; continue; }
    const firstToken = tokens.length;
    const normalized = line.trim().replace(/^\*\s+/, "• ");
    const words = [...normalized.matchAll(/\*\*(\S(?:.*?\S)?)\*\*|\[[^\]]+\]|<Blocker>|\S+/gi)].flatMap(match => {
      const bold = match[1] !== undefined;
      const parts = bold ? match[1].match(/\[[^\]]+\]|<Blocker>|\S+/gi) ?? [] : [match[0]];
      return parts.map(raw => ({ raw, bold }));
    });
    for (const { raw, bold } of words) {
      const badge = raw.toLowerCase() === "<blocker>" || (raw.startsWith("[") && raw.endsWith("]"));
      const badgeText = badge ? raw.slice(1, -1) : raw;
      const blocker = badge && /^blocker$/i.test(badgeText);
      const text = blocker ? "Blocker" : badgeText;
      ctx.font = effectFont(size, badge, outlined, bold);
      const padding = blocker ? 1.12 : outlined ? 1.1 : .6;
      const tokenWidth = ctx.measureText(text).width + (badge ? size * padding : 0);
      const previous = tokens[tokens.length - 1];
      const besideBlocker = blocker || (previous?.badge && previous.text === "Blocker");
      const gap = cursor ? space + (outlined && besideBlocker ? 2 : 0) : 0;
      if (cursor && cursor + gap + tokenWidth > width) { y += lineHeight; cursor = 0; }
      tokens.push({ text, badge, bold, x: x + cursor + (cursor ? gap : 0), y, width: tokenWidth });
      cursor += (cursor ? gap : 0) + tokenWidth;
    }
    // Fit a lone trailing word onto the preceding line with modest horizontal compression.
    const last = tokens[tokens.length - 1];
    const previous = tokens[tokens.length - 2];
    if (outlined && tokens.length - firstToken > 1 && previous.y < last.y && !last.badge) {
      const lastLineStart = tokens.findIndex((token, index) => index >= firstToken && token.y === previous.y);
      const lineTokens = tokens.slice(lastLineStart, -1);
      const lastX = previous.x + previous.width + space;
      const scale = width / (lastX + last.width - x);
      if (scale >= .9 && lineTokens.every(token => !token.badge)) {
        last.x = lastX;
        last.y = previous.y;
        for (const token of [...lineTokens, last]) {
          token.x = x + (token.x - x) * scale;
          token.scale = scale;
        }
        y = previous.y;
      }
    }
    y += lineHeight; cursor = 0;
  }
  return { tokens, height: Math.max(0, y - top - lineHeight) };
}
function drawEffects(text: string, x: number, top: number, width: number, bottom: number, outlined: boolean): boolean {
  let size = outlined ? 32 : 33;
  let layout = effectLayout(text, x, top, width, size, outlined);
  while (layout.height > bottom - top && size > 24) {
    size--; layout = effectLayout(text, x, top, width, size, outlined);
  }
  ctx.save(); ctx.beginPath(); ctx.rect(x - 5, top - size - 5, width + 10, bottom - top + size + 12); ctx.clip();
  ctx.lineJoin = "round";
  for (const token of layout.tokens) {
    ctx.font = effectFont(size, token.badge, outlined, token.bold);
    if (token.badge) {
      const badgeTop = token.y - size * .75;
      const badgeHeight = size * .94;
      if (token.text.toLowerCase() === "blocker") {
        const tip = size * .38;
        const right = token.x + token.width;
        ctx.beginPath(); ctx.moveTo(token.x, badgeTop + badgeHeight / 2);
        ctx.lineTo(token.x + tip, badgeTop); ctx.lineTo(right - tip, badgeTop);
        ctx.lineTo(right, badgeTop + badgeHeight / 2);
        ctx.lineTo(right - tip, badgeTop + badgeHeight); ctx.lineTo(token.x + tip, badgeTop + badgeHeight);
        ctx.closePath(); ctx.fillStyle = "#ec7134"; ctx.fill();
        if (outlined) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 3.5; ctx.stroke(); }
      } else {
        roundRect(token.x, badgeTop, token.width, badgeHeight, 7,
          /once per turn/i.test(token.text) ? "#d83b78" : "#078fc3", outlined ? "#fff" : undefined, 3.5);
      }
      ctx.textAlign = "center";
      ctx.fillStyle = "white"; ctx.fillText(token.text, token.x + token.width / 2, token.y + 2);
      ctx.textAlign = "left";
    } else {
      ctx.save();
      ctx.translate(token.x, token.y + 2);
      if (token.scale) ctx.scale(token.scale, 1);
      ctx.lineWidth = outlined ? 8 : 4; ctx.strokeStyle = "#fff"; ctx.strokeText(token.text, 0, 0);
      ctx.fillStyle = "#11121a"; ctx.fillText(token.text, 0, 0);
      ctx.restore();
    }
  }
  ctx.restore();
  return layout.height <= bottom - top;
}
function drawAttributeBadge(attribute: (typeof attributes)[number], x: number) {
  ctx.save();
  ctx.translate(x, 57);
  // Rounded crown above a flared lower band.
  ctx.beginPath();
  ctx.arc(0, -5, 36, Math.PI - .4, Math.PI * 2 + .4);
  ctx.lineTo(48, 20);
  ctx.quadraticCurveTo(42, 37, 24, 43);
  ctx.quadraticCurveTo(0, 57, -24, 43);
  ctx.quadraticCurveTo(-42, 37, -48, 20);
  ctx.closePath();
  ctx.save();
  ctx.shadowColor = "#0006";
  ctx.shadowBlur = 7;
  ctx.shadowOffsetX = -4;
  ctx.shadowOffsetY = 2;
  ctx.fillStyle = attribute.color; ctx.fill();
  ctx.restore();

  ctx.fillStyle = "#fff"; ctx.textAlign = "center";
  ctx.font = '800 48px "Yu Gothic", "Meiryo", sans-serif';
  ctx.fillText(attribute.kanji, 0, 14);
  const label = [...attribute.label.toUpperCase()];
  const angleStep = Math.min(.32, 1.6 / Math.max(1, label.length - 1));
  ctx.font = `800 ${Math.min(18, 108 / label.length)}px "Arial Narrow", Arial, sans-serif`;
  ctx.textBaseline = "middle";
  for (const [index, letter] of label.entries()) {
    const angle = (index - (label.length - 1) / 2) * angleStep;
    ctx.save();
    ctx.translate(40 * Math.sin(angle), 41 * Math.cos(angle));
    ctx.rotate(-angle);
    if (attribute.label === "Strike") ctx.scale(1, 1.15);
    ctx.fillText(letter, 0, 0);
    ctx.restore();
  }
  ctx.restore();
}

function drawCounterTab(counter: string, frameColor: string, textColor: string, outlined: boolean) {
  if (counter === "0") return;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(43, 444);
  ctx.lineTo(83, 477); ctx.quadraticCurveTo(94, 487, 94, 501);
  ctx.lineTo(94, 789); ctx.quadraticCurveTo(94, 804, 82, 815);
  ctx.lineTo(43, 849);
  ctx.closePath();
  ctx.fillStyle = frameColor; ctx.fill();

  ctx.translate(69, 646);
  ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  ctx.fillStyle = textColor;
  ctx.lineJoin = "round";
  ctx.strokeStyle = frameColor;
  ctx.lineWidth = 7;
  ctx.font = '600 26px "Segoe UI", Arial, sans-serif';
  if (outlined) ctx.strokeText("COUNTER", -116, 0);
  ctx.fillText("COUNTER", -116, 0);
  ctx.font = '700 44px "Segoe UI", Arial, sans-serif';
  if (outlined) ctx.strokeText(`+${counter}`, 16, 0);
  ctx.fillText(`+${counter}`, 16, 0);

  ctx.beginPath();
  ctx.moveTo(-148, -13); ctx.lineTo(-124, -3);
  ctx.lineTo(-134, 1); ctx.lineTo(-122, 13);
  ctx.lineTo(-148, 3); ctx.lineTo(-139, -1);
  ctx.closePath();
  if (outlined) ctx.stroke();
  ctx.fillStyle = textColor === "#fff" ? "#ffe82c" : textColor;
  ctx.fill();
  ctx.restore();
}

function drawCost(cost: string, frameColor: string, textColor: string) {
  ctx.beginPath(); ctx.arc(89, 91, 67, 0, Math.PI * 2);
  ctx.fillStyle = frameColor; ctx.fill();
  ctx.fillStyle = textColor;
  if (cost === "4") {
    ctx.beginPath();
    ctx.moveTo(53, 99); ctx.lineTo(84, 50); ctx.lineTo(108, 50);
    ctx.lineTo(94, 87); ctx.lineTo(111, 67); ctx.lineTo(111, 97);
    ctx.lineTo(121, 97); ctx.lineTo(121, 114); ctx.lineTo(111, 114);
    ctx.lineTo(111, 126); ctx.lineTo(91, 126); ctx.lineTo(91, 114);
    ctx.lineTo(53, 114); ctx.closePath();
    ctx.moveTo(73, 97); ctx.lineTo(90, 72); ctx.lineTo(88, 97);
    ctx.closePath();
    ctx.fill("evenodd");
  } else if (cost === "5") {
    ctx.beginPath();
    ctx.moveTo(65, 50); ctx.lineTo(111, 50); ctx.lineTo(94, 72);
    ctx.lineTo(82, 72); ctx.lineTo(80, 82);
    ctx.quadraticCurveTo(87, 77, 97, 79);
    ctx.bezierCurveTo(129, 80, 128, 126, 99, 133);
    ctx.bezierCurveTo(82, 139, 59, 129, 56, 112);
    ctx.lineTo(75, 108); ctx.quadraticCurveTo(79, 120, 92, 116);
    ctx.bezierCurveTo(105, 112, 104, 94, 92, 94);
    ctx.quadraticCurveTo(85, 93, 79, 100);
    ctx.lineTo(58, 97); ctx.closePath();
    ctx.moveTo(111, 61); ctx.lineTo(119, 50); ctx.lineTo(119, 72);
    ctx.lineTo(102, 72); ctx.closePath();
    ctx.fill();
  } else {
    ctx.font = `800 ${fitted(cost, 101, 94, 800)}px Arial`;
    ctx.fillText(cost, 89, 124);
  }
}

function renderCard() {
  const warning = byId<HTMLElement>("card-warning"); warning.textContent = "";
  const frameColor = value("card-frame-color") as keyof typeof frameColors;
  const frame = frameColors[frameColor] ?? frameColors.blue;
  const frameTextColor = "text" in frame ? frame.text : "#fff";
  const outerBorder = value("card-outer-border");
  const showColoredBorder = value("card-colored-border") === "visible";
  const fullBleedArtwork = outerBorder === "hidden" && !showColoredBorder;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // Geometry is measured from the supplied 1000 × 1398 reference card.
  ctx.save(); ctx.beginPath(); ctx.roundRect(0, 0, 1000, 1398, 43); ctx.clip();
  if (outerBorder !== "hidden") {
    ctx.fillStyle = "#fdfdfd"; ctx.fillRect(0, 0, 1000, 1398);
  }
  const borderImage = outerBorderAssets.find(asset => asset.value === outerBorder)?.image;
  if (borderImage?.naturalWidth) {
    ctx.drawImage(borderImage, 0, 0, 1000, 1398);
  }
  ctx.save();
  if (!fullBleedArtwork) {
    traceArtworkOutline(); ctx.clip();
  }
  if (artwork) {
    drawCover(artwork, 0, 0, 1000, 1398);
  } else {
    const backing = ctx.createLinearGradient(0, 43, 1000, 1358);
    backing.addColorStop(0, frame.backing[0]); backing.addColorStop(1, frame.backing[1]);
    ctx.fillStyle = backing; ctx.fillRect(0, 0, 1000, 1398);
    ctx.fillStyle = "#ffffff24"; ctx.fillRect(51, 50, 898, 1148);
    ctx.fillStyle = "#ffffffb5"; ctx.font = "600 28px Arial"; ctx.textAlign = "center";
    ctx.fillText("Choose artwork to complete the card", 500, 535); ctx.textAlign = "left";
  }
  ctx.restore();

  // The footer fill is independent of the frame rails.
  ctx.beginPath(); ctx.moveTo(956, 1188);
  traceFooterTop();
  traceFooterOutside();
  ctx.closePath();
  ctx.fillStyle = frame.rail; ctx.fill();
  if (showColoredBorder) {
    traceArtworkOutline();
    ctx.moveTo(43, 1188); traceFooterOutside(); ctx.lineTo(956, 1188);
    ctx.save(); ctx.lineJoin = "round";
    ctx.strokeStyle = frame.rail; ctx.lineWidth = 8; ctx.stroke();
    ctx.restore();
    // Keep a crisp corner where the top rail turns into the slope.
    ctx.beginPath();
    ctx.moveTo(75, 43);
    tracePowerShoulder();
    ctx.lineTo(664, 118);
    ctx.bezierCurveTo(647, 118, 635, 124, 621, 103);
    ctx.lineTo(588, 50);
    ctx.lineTo(75, 50);
    ctx.closePath();
    ctx.fillStyle = frame.rail; ctx.fill();
  }
  drawCounterTab(value("card-counter"), frame.rail, frameTextColor, byId<HTMLInputElement>("card-counter-outline").checked);
  const effectBackground = value("card-effect-background");
  const outlinedEffects = effectBackground === "artwork";
  if (effectBackground === "panel") roundRect(63, 850, 874, 300, 7, "#e8edf0a8", "#ffffff54", 1);
  const effectFits = drawEffects(value("card-effect"), 74, outlinedEffects ? 904 : 894, outlinedEffects ? 819 : 860, 1134, outlinedEffects);
  if (!effectFits) warning.textContent = "Effect text is crowded. Shorten it or check the preview before exporting.";
  ctx.textAlign = "center"; ctx.fillStyle = frameTextColor;
  ctx.save();
  ctx.translate(500, 1198);
  ctx.scale(.68, 1);
  ctx.font = '700 30px Roboto, "Segoe UI", sans-serif';
  ctx.letterSpacing = "13px";
  ctx.fillText("CHARACTER", 0, 0);
  ctx.restore();
  const name = value("card-name") || "Card name"; ctx.font = `700 ${fitted(name, 845, 62)}px Arial, sans-serif`;
  ctx.fillText(name, 500, 1275);
  const cardType = value("card-type");
  ctx.font = "400 29px Arial, sans-serif";
  const capsuleWidth = Math.min(550, Math.max(390, Math.ceil(ctx.measureText(cardType).width) + 48));
  roundRect(500 - capsuleWidth / 2, 1294, capsuleWidth, 45, 23, frame.capsule.fill, frame.capsule.edge, frameColor === "black" ? 2 : 4);
  ctx.fillStyle = frameTextColor;
  ctx.font = `400 ${fitted(cardType, capsuleWidth - 48, 29, 400)}px Arial, sans-serif`;
  ctx.fillText(cardType, 500, 1327, capsuleWidth - 48);
  // Lower-left color emblem and right-side collection metadata.
  ctx.beginPath(); ctx.moveTo(49, 1282);
  ctx.arcTo(74, 1269, 124, 1295, 9);
  ctx.arcTo(124, 1295, 124, 1356, 9);
  ctx.arcTo(124, 1356, 74, 1381, 9);
  ctx.arcTo(74, 1381, 24, 1356, 9);
  ctx.arcTo(24, 1356, 24, 1295, 9);
  ctx.arcTo(24, 1295, 74, 1269, 9);
  ctx.closePath();
  ctx.fillStyle = frame.emblem.fill; ctx.fill();
  ctx.lineWidth = 5; ctx.strokeStyle = frame.emblem.highlight; ctx.stroke();
  const vertices = [[74, 1282], [111, 1302], [111, 1348], [74, 1368], [37, 1348], [37, 1302]] as const;
  const section = frame.emblem.filledSection;
  if (section !== null) {
    ctx.beginPath(); ctx.moveTo(74, 1325);
    const [startX, startY] = vertices[section];
    const [endX, endY] = vertices[(section + 1) % vertices.length];
    ctx.lineTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.closePath();
    ctx.fillStyle = "sectionColor" in frame.emblem ? frame.emblem.sectionColor : frame.rail; ctx.fill();
  }
  ctx.beginPath(); ctx.moveTo(...vertices[0]);
  for (const [x, y] of vertices.slice(1)) ctx.lineTo(x, y);
  ctx.closePath();
  ctx.strokeStyle = frame.emblem.edge; ctx.lineWidth = 2.5; ctx.stroke();
  ctx.beginPath(); ctx.moveTo(74, 1282); ctx.lineTo(74, 1368); ctx.moveTo(37, 1302);
  ctx.lineTo(111, 1348); ctx.moveTo(111, 1302); ctx.lineTo(37, 1348); ctx.stroke();
  ctx.beginPath(); ctx.arc(74, 1325, 8, 0, Math.PI * 2);
  ctx.fillStyle = frame.emblem.edge; ctx.fill();
  ctx.textAlign = "right"; ctx.font = '600 23px "Arial Narrow", Arial, sans-serif'; ctx.fillStyle = frameTextColor;
  ctx.fillText(value("card-number"), 874, 1342);
  roundRect(878, 1323, 27, 21, 2, "#fff"); ctx.fillStyle = "#161b24";
  ctx.textAlign = "center"; ctx.font = '700 17px "Arial Narrow", Arial, sans-serif'; ctx.fillText(value("card-rarity"), 892, 1340, 23);
  ctx.beginPath(); ctx.moveTo(917, 1311);
  ctx.bezierCurveTo(922, 1318, 931, 1302, 939, 1305);
  ctx.bezierCurveTo(946, 1308, 941, 1314, 937, 1312);
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke();
  ctx.beginPath(); ctx.arc(927, 1331, 18, 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill();
  ctx.fillStyle = "#101218"; ctx.font = "800 22px Arial";
  ctx.fillText(value("card-block-number"), 927, 1339, 30);
  // Upper stat labels sit over the white header.
  ctx.fillStyle = "#07080a"; ctx.textAlign = "center";
  const power = value("card-power") || "0";
  const twoAttributes = selectedAttributes.size === maxAttributes;
  ctx.font = `700 ${fitted(power, twoAttributes ? 130 : 210, 90) * .98}px Arial`;
  const powerX = twoAttributes ? 707 : 756;
  ctx.save(); ctx.lineJoin = "round"; ctx.lineWidth = 9; ctx.strokeStyle = "#fff";
  ctx.strokeText(power, powerX, 96);
  ctx.restore();
  ctx.fillText(power, powerX, 96);
  [...selectedAttributes].forEach((attribute, index) => {
    drawAttributeBadge(attribute, 921 - index * 100);
  });
  drawCost(value("card-cost") || "0", frame.rail, frameTextColor);
  ctx.restore();
}

function cardField(id: string) {
  return byId<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>(id);
}

function downloadCard(href: string, extension: string) {
  const link = document.createElement("a");
  link.href = href;
  link.download = `${(value("card-number") || "proxykit-card").replace(/[^a-z0-9_-]/gi, "_")}.${extension}`;
  link.click();
}

function artworkDataUrl(blob: Blob) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

export function initCardGenerator() {
  cardFields.forEach(id => cardField(id).addEventListener("input", renderCard));
  const counterOutline = byId<HTMLInputElement>("card-counter-outline");
  counterOutline.addEventListener("change", renderCard);
  const options = byId("card-attribute-options");
  const checkboxes = attributes.map(attribute => {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = selectedAttributes.has(attribute);
    label.append(checkbox, attribute.label);
    options.append(label);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked && selectedAttributes.size < maxAttributes) {
        selectedAttributes.add(attribute);
      } else if (!checkbox.checked && selectedAttributes.size > 1) {
        selectedAttributes.delete(attribute);
      }
      syncAttributeOptions();
      renderCard();
    });
    return checkbox;
  });
  function syncAttributeOptions() {
    checkboxes.forEach((checkbox, index) => {
      checkbox.checked = selectedAttributes.has(attributes[index]);
      checkbox.disabled = checkbox.checked
        ? selectedAttributes.size === 1
        : selectedAttributes.size === maxAttributes;
    });
  }
  syncAttributeOptions();
  const warning = byId("card-warning");
  const artworkInput = byId<HTMLInputElement>("card-artwork");
  artworkInput.addEventListener("change", async () => {
    const file = artworkInput.files?.[0];
    if (!file) return;
    const revision = ++loadRevision;
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.src = url;
    try {
      await image.decode();
      if (revision !== loadRevision) return;
      if (artworkUrl) URL.revokeObjectURL(artworkUrl);
      artworkUrl = url;
      artwork = image;
      renderCard();
    } catch {
      if (revision === loadRevision) warning.textContent = "Could not open that artwork image.";
    } finally {
      if (artworkUrl !== url) URL.revokeObjectURL(url);
    }
  });
  byId("card-export").addEventListener("click", async () => {
    const fields = Object.fromEntries(cardFields.map(id => [id, cardField(id).value]));
    const labels = [...selectedAttributes].map(attribute => attribute.label);
    const showCounterOutline = counterOutline.checked;
    try {
      const source = artworkUrl
        ? await artworkDataUrl(await (await fetch(artworkUrl)).blob())
        : artwork?.src ?? null;
      const json = JSON.stringify({ version: 1, fields, attributes: labels, artwork: source, counterOutline: showCounterOutline }, null, 2);
      const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
      downloadCard(url, "proxykit.json");
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch {
      warning.textContent = "Could not export this card.";
    }
  });
  const importFile = byId<HTMLInputElement>("card-import-file");
  byId("card-import").addEventListener("click", () => importFile.click());
  importFile.addEventListener("change", async () => {
    const file = importFile.files?.[0];
    importFile.value = "";
    if (!file) return;
    const revision = ++loadRevision;
    try {
      const card = JSON.parse(await file.text());
      if (card?.version !== 1 || !card.fields || !Array.isArray(card.attributes)) {
        throw new Error("Choose a card file exported by ProxyKit.");
      }
      if (card.counterOutline !== undefined && typeof card.counterOutline !== "boolean") {
        throw new Error("The card file contains an invalid counter outline setting.");
      }
      const fields = cardFields.map(id => {
        const field = cardField(id);
        const fieldValue = card.fields[id] === undefined ? borderDefaults[id] : card.fields[id];
        if (typeof fieldValue !== "string"
          || (field instanceof HTMLSelectElement && ![...field.options].some(option => option.value === fieldValue))
          || (field instanceof HTMLInputElement && field.type === "number" && fieldValue !== ""
            && !/^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?$/i.test(fieldValue))) {
          throw new Error("The card file contains invalid field values.");
        }
        return { field, fieldValue };
      });
      const selection = card.attributes.map((label: unknown) => attributes.find(attribute => attribute.label === label));
      if (selection.length < 1 || selection.length > maxAttributes
        || new Set(selection).size !== selection.length || selection.includes(undefined)) {
        throw new Error("The card file contains invalid attributes.");
      }
      let image = null;
      if (card.artwork !== null) {
        if (typeof card.artwork !== "string" || !/^data:image\/(png|jpeg|webp);base64,/i.test(card.artwork)) {
          throw new Error("The card file contains unsupported artwork.");
        }
        image = new Image();
        image.src = card.artwork;
        await image.decode();
      }
      if (revision !== loadRevision) return;
      fields.forEach(({ field, fieldValue }) => { field.value = fieldValue; });
      counterOutline.checked = card.counterOutline ?? true;
      selectedAttributes.clear();
      for (const attribute of selection) selectedAttributes.add(attribute);
      syncAttributeOptions();
      if (artworkUrl) URL.revokeObjectURL(artworkUrl);
      artworkUrl = null;
      artwork = image;
      artworkInput.value = "";
      renderCard();
    } catch (error) {
      if (revision === loadRevision) warning.textContent = error instanceof SyntaxError
        ? "Choose a valid ProxyKit card file."
        : error instanceof Error ? error.message : "Could not import this card.";
    }
  });
  const effectFontsReady = Promise.all([
    document.fonts.load("400 32px Roboto"),
    document.fonts.load("450 33px Roboto"),
    document.fonts.load("500 32px Roboto"),
    document.fonts.load("700 32px Roboto"),
  ]).catch(error => console.error("Could not load the bundled effect font.", error));
  const borderImagesReady = Promise.all(outerBorderAssets.map(({ path, image }) => {
    image.src = path;
    return image.decode().then(renderCard).catch(error => {
      console.error(`Could not load border image ${path}.`, error);
    });
  }));
  byId<HTMLButtonElement>("card-download").addEventListener("click", async () => {
    await Promise.all([effectFontsReady, borderImagesReady]);
    renderCard();
    downloadCard(canvas.toDataURL("image/png"), "png");
  });
  renderCard();
  void effectFontsReady.then(renderCard);
}
