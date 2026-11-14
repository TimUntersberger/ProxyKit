const byId = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = byId<HTMLCanvasElement>("card-canvas");
const ctx = canvas.getContext("2d")!;
let artwork: HTMLImageElement | null = null;
let artworkUrl: string | null = null;

const value = (id: string) => (byId<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>(id).value || "").trim();

function roundRect(x: number, y: number, w: number, h: number, r: number, fill: string, stroke?: string, lineWidth = 1) {
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
type EffectToken = { text: string; badge: boolean; x: number; y: number; width: number };
function effectLayout(source: string, x: number, top: number, width: number, size: number) {
  ctx.font = `700 ${size}px Arial, sans-serif`;
  const lineHeight = size * 1.34;
  const space = ctx.measureText(" ").width;
  const tokens: EffectToken[] = [];
  let y = top, cursor = 0;
  const lines = source.replace(/\r/g, "").split("\n");
  for (const line of lines) {
    if (!line.trim()) { y += lineHeight * .4; continue; }
    const normalized = line.trim().replace(/^\*\s*/, "• ");
    for (const match of normalized.matchAll(/\[[^\]]+\]|\S+/g)) {
      const raw = match[0];
      const badge = raw.startsWith("[") && raw.endsWith("]");
      const text = badge ? raw.slice(1, -1) : raw;
      const tokenWidth = ctx.measureText(text).width + (badge ? 22 : 0);
      const gap = cursor ? (badge ? 10 : space) : 0;
      if (cursor && cursor + gap + tokenWidth > width) { y += lineHeight; cursor = 0; }
      tokens.push({ text, badge, x: x + cursor + (cursor ? gap : 0), y, width: tokenWidth });
      cursor += (cursor ? gap : 0) + tokenWidth;
    }
    y += lineHeight; cursor = 0;
  }
  return { tokens, height: Math.max(0, y - top - lineHeight) };
}
function drawEffects(text: string, x: number, top: number, width: number, bottom: number): boolean {
  let size = 33;
  let layout = effectLayout(text, x, top, width, size);
  while (layout.height > bottom - top && size > 24) {
    size--; layout = effectLayout(text, x, top, width, size);
  }
  ctx.save(); ctx.beginPath(); ctx.rect(x - 4, top - size - 2, width + 8, bottom - top + size + 4); ctx.clip();
  ctx.font = `700 ${size}px Arial, sans-serif`;
  for (const token of layout.tokens) {
    if (token.badge) {
      roundRect(token.x, token.y - size + 2, token.width, size + 9, 7,
        /once per turn/i.test(token.text) ? "#d83b78" : "#078fc3");
      ctx.fillStyle = "white"; ctx.fillText(token.text, token.x + 11, token.y + 2);
    } else {
      ctx.lineWidth = 4; ctx.strokeStyle = "#fff"; ctx.strokeText(token.text, token.x, token.y + 2);
      ctx.fillStyle = "#11121a"; ctx.fillText(token.text, token.x, token.y + 2);
    }
  }
  ctx.restore();
  return layout.height <= bottom - top;
}
function renderCard() {
  const warning = byId<HTMLElement>("card-warning"); warning.textContent = "";
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // Geometry is measured from the supplied 1000 × 1398 reference card.
  roundRect(0, 0, 1000, 1398, 43, "#fdfdfd");
  ctx.save(); ctx.beginPath(); ctx.roundRect(42, 43, 916, 1315, 16); ctx.clip();
  const backing = ctx.createLinearGradient(0, 43, 1000, 1358);
  backing.addColorStop(0, "#058dc3"); backing.addColorStop(1, "#0067ad");
  ctx.fillStyle = backing; ctx.fillRect(42, 43, 916, 1315);
  if (artwork) drawCover(artwork, 51, 50, 898, 1148);
  else {
    ctx.fillStyle = "#ffffff24"; ctx.fillRect(51, 50, 898, 1148);
    ctx.fillStyle = "#ffffffb5"; ctx.font = "600 28px Arial"; ctx.textAlign = "center";
    ctx.fillText("Choose artwork to complete the card", 500, 535); ctx.textAlign = "left";
  }
  ctx.restore();
  // Blue artwork rail, including the angled opening beneath the power header.
  ctx.beginPath(); ctx.moveTo(43, 840); ctx.lineTo(43, 51); ctx.quadraticCurveTo(43, 43, 52, 43);
  ctx.lineTo(594, 43); ctx.quadraticCurveTo(611, 43, 619, 57);
  ctx.lineTo(644, 102); ctx.quadraticCurveTo(650, 114, 666, 114);
  ctx.lineTo(948, 114); ctx.quadraticCurveTo(956, 114, 956, 123);
  ctx.lineTo(956, 1190); ctx.quadraticCurveTo(956, 1197, 947, 1197);
  ctx.lineTo(52, 1197); ctx.quadraticCurveTo(43, 1197, 43, 1188); ctx.closePath();
  ctx.strokeStyle = "#006d9c"; ctx.lineWidth = 15; ctx.shadowColor = "#1838488a"; ctx.shadowBlur = 9; ctx.stroke(); ctx.shadowBlur = 0;
  ctx.strokeStyle = "#0793c8"; ctx.lineWidth = 7; ctx.stroke();
  // Translucent rules panel: bounds x63–937, y850–1150.
  roundRect(63, 850, 874, 300, 7, "#e8edf0a8", "#ffffff54", 1);
  const effectFits = drawEffects(value("card-effect"), 74, 894, 850, 1134);
  if (!effectFits) warning.textContent = "Effect text is crowded. Shorten it or check the preview before exporting.";
  // Footer rises to a small class tab at y1167 and ends at y1358.
  const footer = ctx.createLinearGradient(0, 1167, 980, 1358);
  footer.addColorStop(0, "#058fc4"); footer.addColorStop(.5, "#047fbb"); footer.addColorStop(1, "#005fae");
  ctx.beginPath(); ctx.moveTo(43, 1166); ctx.lineTo(52, 1197); ctx.lineTo(325, 1197);
  ctx.lineTo(352, 1167); ctx.lineTo(646, 1167); ctx.lineTo(674, 1197);
  ctx.lineTo(949, 1197); ctx.lineTo(957, 1166); ctx.lineTo(957, 1348);
  ctx.quadraticCurveTo(957, 1358, 947, 1358); ctx.lineTo(53, 1358);
  ctx.quadraticCurveTo(43, 1358, 43, 1348); ctx.closePath();
  ctx.fillStyle = footer; ctx.fill(); ctx.strokeStyle = "#006393"; ctx.lineWidth = 5; ctx.stroke();
  ctx.textAlign = "center"; ctx.fillStyle = "#fff";
  const kind = "CHARACTER"; ctx.font = "700 23px Impact, Arial Narrow, sans-serif";
  ctx.fillText(kind.split("").join("  "), 500, 1198);
  const name = value("card-name") || "Card name"; ctx.font = `700 ${fitted(name, 845, 62)}px Arial, sans-serif`;
  ctx.fillText(name, 500, 1275);
  roundRect(305, 1294, 390, 45, 23, "#006da9", "#005783", 4);
  ctx.font = "400 29px Arial, sans-serif"; ctx.fillText(value("card-type"), 500, 1327);
  // Lower-left color emblem and right-side collection metadata.
  ctx.beginPath(); ctx.moveTo(74, 1269); ctx.lineTo(124, 1295); ctx.lineTo(124, 1356);
  ctx.lineTo(74, 1381); ctx.lineTo(24, 1356); ctx.lineTo(24, 1295); ctx.closePath();
  ctx.fillStyle = "#006eaa"; ctx.shadowColor = "#14394c99"; ctx.shadowBlur = 10; ctx.fill(); ctx.shadowBlur = 0;
  ctx.lineWidth = 7; ctx.strokeStyle = "#009bd0"; ctx.stroke();
  ctx.beginPath(); ctx.moveTo(74, 1282); ctx.lineTo(111, 1302); ctx.lineTo(111, 1348);
  ctx.lineTo(74, 1368); ctx.lineTo(37, 1348); ctx.lineTo(37, 1302); ctx.closePath();
  ctx.strokeStyle = "#004b7d"; ctx.lineWidth = 4; ctx.stroke();
  ctx.beginPath(); ctx.moveTo(74, 1282); ctx.lineTo(74, 1368); ctx.moveTo(37, 1302);
  ctx.lineTo(111, 1348); ctx.moveTo(111, 1302); ctx.lineTo(37, 1348); ctx.stroke();
  ctx.textAlign = "right"; ctx.font = "700 20px Arial"; ctx.fillStyle = "#fff";
  ctx.fillText(value("card-number"), 874, 1342);
  roundRect(879, 1325, 27, 20, 1, "#fff"); ctx.fillStyle = "#161b24";
  ctx.textAlign = "center"; ctx.font = "700 15px Arial"; ctx.fillText(value("card-rarity"), 892, 1341);
  ctx.beginPath(); ctx.arc(927, 1331, 18, 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill();
  ctx.fillStyle = "#101218"; ctx.font = "800 22px Arial"; ctx.fillText(value("card-cost"), 927, 1339);
  // Upper stat field follows the reference's white angled edge.
  ctx.beginPath(); ctx.moveTo(584, 0); ctx.lineTo(1000, 0); ctx.lineTo(1000, 118);
  ctx.lineTo(659, 118); ctx.quadraticCurveTo(637, 118, 627, 102);
  ctx.lineTo(604, 56); ctx.quadraticCurveTo(597, 44, 584, 44); ctx.closePath();
  ctx.fillStyle = "#fdfdfd"; ctx.fill();
  ctx.beginPath(); ctx.moveTo(584, 45); ctx.quadraticCurveTo(606, 45, 614, 63);
  ctx.lineTo(638, 104); ctx.quadraticCurveTo(647, 115, 659, 115); ctx.lineTo(951, 115);
  ctx.strokeStyle = "#007bab"; ctx.lineWidth = 8; ctx.stroke();
  ctx.fillStyle = "#07080a"; ctx.textAlign = "center";
  const power = value("card-power") || "0"; ctx.font = `700 ${fitted(power, 224, 86, 700)}px Arial`;
  ctx.fillText(power, 756, 96);
  ctx.beginPath(); ctx.arc(921, 57, 42, 0, Math.PI * 2); ctx.fillStyle = "#722357"; ctx.fill();
  ctx.fillStyle = "#fff"; ctx.font = "700 40px Arial"; ctx.fillText("特", 921, 69);
  ctx.font = "700 16px Arial"; ctx.fillText("SPECIAL", 921, 92);
  const costGradient = ctx.createRadialGradient(80, 80, 8, 88, 90, 70);
  costGradient.addColorStop(0, "#05a6d3"); costGradient.addColorStop(1, "#006da9");
  ctx.beginPath(); ctx.arc(89, 91, 67, 0, Math.PI * 2); ctx.fillStyle = costGradient;
  ctx.shadowColor = "#1c394f9c"; ctx.shadowBlur = 14; ctx.fill(); ctx.shadowBlur = 0;
  ctx.fillStyle = "white"; ctx.font = `800 ${fitted(value("card-cost"), 96, 84, 800)}px Arial`;
  ctx.fillText(value("card-cost") || "0", 89, 119);
  ctx.textAlign = "left";
}

export function initCardGenerator() {
  document.querySelectorAll<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>(".card-form input, .card-form textarea, .card-form select")
    .forEach(input => { if (input.id !== "card-artwork") input.addEventListener("input", renderCard); });
  byId<HTMLInputElement>("card-artwork").addEventListener("change", event => {
    const file = (event.currentTarget as HTMLInputElement).files?.[0]; if (!file) return;
    if (artworkUrl) URL.revokeObjectURL(artworkUrl);
    artworkUrl = URL.createObjectURL(file); const image = new Image();
    image.onload = () => { artwork = image; renderCard(); };
    image.onerror = () => { byId<HTMLElement>("card-warning").textContent = "Could not open that artwork image."; };
    image.src = artworkUrl;
  });
  byId<HTMLButtonElement>("card-download").addEventListener("click", () => {
    renderCard(); const link = document.createElement("a");
    link.href = canvas.toDataURL("image/png");
    link.download = `${(value("card-number") || "proxykit-card").replace(/[^a-z0-9_-]/gi, "_")}.png`;
    link.click();
  });
  renderCard();
}
