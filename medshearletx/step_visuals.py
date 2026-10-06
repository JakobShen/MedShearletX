"""Local iteration previews, independent of model calls and final evaluation."""

import html
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from .explainer import to_image


def _atomic_write(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".step-", suffix=path.suffix, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class StepVisualizer:
    """Callable ``(current_mask, history)`` that saves every iteration locally.

    History can be the current record or a list ending with that record. The
    supplied mask must be the current iteration's mask, rather than a selected
    best mask whose objective differs from that history record.
    """

    def __init__(self, output_dir, input_image, transform, coefficients, *,
                 model, target_label, normalize_final, grid_size, mask_resolution="grid"):
        if not isinstance(input_image, Image.Image):
            raise TypeError("input_image must be a PIL image")
        self.output = Path(output_dir).expanduser().resolve()
        self.input_image = input_image.convert("RGB").copy()
        self.transform = transform
        self.coefficients = np.asarray(coefficients)
        if self.coefficients.ndim != 4 or not np.isfinite(self.coefficients).all():
            raise ValueError("coefficients must be finite C,K,H,W")
        channels, self.bands, height, width = self.coefficients.shape
        if channels != 3 or self.input_image.size != (width, height):
            raise ValueError("coefficients and RGB input image dimensions must agree")
        if mask_resolution not in {"grid", "full"}:
            raise ValueError("mask_resolution must be grid or full")
        if type(grid_size) is not int or grid_size < 1 or (
            mask_resolution == "grid" and grid_size > min(height, width)
        ):
            raise ValueError("grid_size must be a positive integer within image dimensions")
        if type(normalize_final) is not bool:
            raise ValueError("normalize_final must be a boolean")
        self.grid_size, self.normalize_final = grid_size, normalize_final
        self.mask_resolution = mask_resolution
        self.mask_shape = (self.bands, height, width) if mask_resolution == "full" else (self.bands, grid_size, grid_size)
        if mask_resolution == "grid":
            self.y_index = np.arange(height) * grid_size // height
            self.x_index = np.arange(width) * grid_size // width
        self.model, self.target_label = str(model), str(target_label)
        self.records = {}
        self.final_summary = None
        _atomic_write(self.output / "input.png", lambda path: self.input_image.save(path, format="PNG"))

    def __call__(self, mask, history):
        record = dict(history[-1] if isinstance(history, (list, tuple)) else history)
        step = record.get("step")
        if type(step) is not int or step < 0:
            raise ValueError("history must contain a nonnegative integer step")
        mask = np.asarray(mask)
        if mask.shape != self.mask_shape or not np.isfinite(mask).all():
            raise ValueError(f"mask must be finite with configured {self.mask_resolution} dimensions {self.mask_shape}")
        if np.any((mask < 0) | (mask > 1)):
            raise ValueError("mask values must be in [0,1]")
        dense = mask[None] if self.mask_resolution == "full" else mask[:, self.y_index[:, None], self.x_index[None, :]][None]
        kept = np.clip(self.transform.decode(self.coefficients * dense), 0, 1)
        if self.normalize_final and kept.max() > 0:
            kept = kept / kept.max()
        removed = self.transform.decode(self.coefficients * (1 - dense))
        kept_image, removed_image = to_image(kept), to_image(removed)
        stem = f"step{step:03d}"
        relative = {"kept": f"images/f_n/{stem}.png", "removed": f"images/removed/{stem}.png",
                    "mask": f"masks/{stem}.npy", "figure": f"figures/f_n/{stem}.png"}
        for name, image in (("kept", kept_image), ("removed", removed_image)):
            _atomic_write(self.output / relative[name], lambda path, image=image: image.save(path, format="PNG"))
        _atomic_write(self.output / relative["mask"], lambda path: np.save(path, mask))
        record.update(preview_mask_mean=float(np.mean(dense)), preview=relative)
        self._save_figure(kept_image, removed_image, record, self.output / relative["figure"])
        self.records[step] = record
        self.final_summary = None
        self._write_index()
        return {name: str(self.output / path) for name, path in relative.items()}

    def finalize(self, result):
        """Add the independently evaluated selected mask to the local index."""
        summaries = []
        for key, name in (("reference", "Original"), ("retained", "Final kept"), ("removed", "Final removed")):
            evidence = result[key]
            summaries.append({
                "image": name, "count": evidence.get("sample_counts", {}).get(self.target_label),
                "samples": evidence["requests"], "frequency": evidence["target_score"],
                "interval": evidence.get("diagnostics", {}).get("sample_wilson_95", {}).get(self.target_label),
            })
        links = {"comparison": "comparison.png", "result": "result.json"}
        if (self.output / "optimization.png").is_file():
            links["optimization"] = "optimization.png"
        self.final_summary = {
            "selected_step": result.get("optimization_diagnostics", {}).get("selected_step"),
            "retained_frequency_ratio": result.get("retained_frequency_ratio"),
            "samples": summaries, "links": links,
        }
        self._write_index()

    def _write_index(self):
        data = {"model": self.model, "target_label": self.target_label,
                "mask_resolution": self.mask_resolution,
                "note": "Current-iteration previews; displayed images have no per-step VLM evaluation.",
                "steps": [self.records[key] for key in sorted(self.records)]}
        if self.final_summary is not None:
            data["final"] = self.final_summary
        encoded = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
        _atomic_write(self.output / "metrics.json", lambda path: path.write_text(encoded + "\n", encoding="utf-8"))
        page = _INDEX.replace("@TITLE@", html.escape(self.model)).replace("@TARGET@", html.escape(self.target_label))
        page = page.replace("@DATA@", encoded.replace("<", "\\u003c"))
        _atomic_write(self.output / "index.html", lambda path: path.write_text(page, encoding="utf-8"))

    def _save_figure(self, kept, removed, record, path):
        from matplotlib import rc_context
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        loss = record.get("loss")
        loss_text = f"{loss:.4f}" if loss is not None else "unavailable"
        with rc_context({"font.family": "DejaVu Serif"}):
            figure = Figure(figsize=(12, 5), dpi=150, facecolor="white")
            FigureCanvasAgg(figure)
            figure.suptitle(f"ShearletX — {self.model}\nStep {record['step']} · loss {loss_text} · "
                           f"mask mean {record['preview_mask_mean']:.4f}", fontsize=16, y=0.97)
            for index, (image, title) in enumerate(((self.input_image, "Original"), (kept, "Current kept image"),
                                                    (removed, "Current removed image"))):
                axis = figure.add_axes([0.01 + index / 3, 0.15, 0.313, 0.67])
                axis.imshow(image, interpolation="nearest")
                axis.set_title(title, fontsize=11, pad=6)
                axis.set_axis_off()
            figure.text(0.025, 0.075, f"Fixed target: {self.target_label}", fontsize=9)
            figure.text(0.025, 0.035,
                        "Iteration preview: loss uses noisy optimization probes. Displayed images have no per-step VLM score; "
                        "final selected-mask evaluation is separate.", fontsize=7.5)
            _atomic_write(path, lambda temporary: figure.savefig(temporary, format="png", dpi=150))
            figure.clear()


_INDEX = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ShearletX iteration previews</title>
<style>
body{font:16px system-ui,sans-serif;margin:0;background:#f4f5f7;color:#20252b}
main{max-width:1200px;margin:32px auto;padding:24px;background:white;border-radius:16px}
h1{font:28px Georgia,serif;margin:0 0 10px}p{line-height:1.5;color:#52606b}
.controls{display:flex;gap:12px;align-items:center;margin:24px 0}input{flex:1}
button,a{color:#1c4c79}button{padding:6px 12px;background:#fff;border:1px solid #bdc7d1;border-radius:6px}
img{width:100%;display:block}#metrics{font:13px ui-monospace,monospace;white-space:pre-wrap;background:#f4f5f7;padding:16px}
nav{display:flex;gap:18px;margin:12px 0}label{font-variant-numeric:tabular-nums;min-width:90px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce2e8}
#final{padding:18px;background:#f4f7fa;border-radius:12px;margin-top:22px}
</style>
<main><h1>ShearletX · @TITLE@</h1><p>Fixed target: <strong>@TARGET@</strong>.<br>
Current-iteration previews use the supplied mask. These images have no per-step measured class probability.
Final selected-mask validation is saved separately. Refresh this page to load newly saved steps.</p>
<section id="final" hidden><h2>Final selected-mask evaluation</h2><p id="selected"></p>
<table><thead><tr><th>Image</th><th>Target responses</th><th>Frequency</th><th>95% Wilson interval</th></tr></thead>
<tbody id="final-samples"></tbody></table><p id="retained-ratio"></p><nav id="final-links"></nav></section>
<div class="controls"><button id="previous">Previous</button><input id="step" type="range" min="0" value="0">
<button id="next">Next</button><label id="position"></label></div>
<img id="figure" alt="Original, current kept image, and current removed image">
<nav><a id="kept">Kept PNG</a><a id="removed">Removed PNG</a><a id="mask">Mask NPY</a><a href="metrics.json">All metrics</a></nav>
<pre id="metrics"></pre></main>
<script>
const data=@DATA@, rows=data.steps, slider=document.getElementById('step');
if(data.final){const final=data.final;document.getElementById('final').hidden=false;
document.getElementById('selected').textContent='Selected mask: iteration '+(final.selected_step??'unavailable')+
'. These independent samples evaluate the final selected mask, which may differ from the last iteration preview.';
for(const row of final.samples){const tr=document.createElement('tr');
const values=[row.image,row.count==null?'unavailable':row.count+'/'+row.samples,(100*row.frequency).toFixed(2)+'%',
row.interval?row.interval.map(x=>(100*x).toFixed(1)+'%').join('–'):'unavailable'];
for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}
document.getElementById('final-samples').append(tr);}
document.getElementById('retained-ratio').textContent='Retained sampling frequency ratio: '+
(final.retained_frequency_ratio==null?'undefined':(100*final.retained_frequency_ratio).toFixed(2)+'%');
for(const [name,path] of Object.entries(final.links)){const a=document.createElement('a');a.href=path;
a.textContent={comparison:'Final comparison',result:'Complete result JSON',optimization:'Optimization trace'}[name];
document.getElementById('final-links').append(a);}}
slider.max=Math.max(0,rows.length-1);slider.value=slider.max;
function show(){const row=rows[Number(slider.value)];if(!row)return;
document.getElementById('position').textContent='Step '+row.step;
document.getElementById('figure').src=row.preview.figure;
for(const name of ['kept','removed','mask'])document.getElementById(name).href=row.preview[name];
const metrics={...row};delete metrics.preview;document.getElementById('metrics').textContent=JSON.stringify(metrics,null,2);}
slider.addEventListener('input',show);
document.getElementById('previous').onclick=()=>{slider.value=Math.max(0,Number(slider.value)-1);show();};
document.getElementById('next').onclick=()=>{slider.value=Math.min(rows.length-1,Number(slider.value)+1);show();};
show();
</script></html>
"""
