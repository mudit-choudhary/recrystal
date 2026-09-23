# Recrystal

A layout parser for research papers: PDF in, ordered typed blocks out.

```
PDF ──▶ [detector: fine-tuned YOLO11, 12 classes] ──▶ regions
    ──▶ [assembler: textreflow] ──▶ typed blocks ──▶ [grain_growth] ──▶ chunks
```

A YOLO11 layout model fine-tuned on research papers (DocLayNet's 11 classes
plus `Authors`) finds the regions on each page, run through onnxruntime with no
ultralytics at runtime. PyMuPDF supplies the words, each assigned to the
smallest region containing it; regions the model merged across a column gutter
are split back apart. [textreflow](https://github.com/mudit-choudhary/textreflow)
then rebuilds reading order and paragraphs.

## Install

```
pip install recrystal            # CPU
pip install recrystal[gpu]       # CUDA 12 / cuDNN 9, via onnxruntime-gpu
```

`onnxruntime` and `onnxruntime-gpu` share one package directory, so whichever
is installed last wins. If another package pulls in the CPU build after the GPU
one (chromadb does), reinstall `onnxruntime-gpu`. A GPU session that silently
falls back to CPU is logged as a warning.

## Use

```python
from recrystal import parse_pdf

blocks, dropped = parse_pdf("paper.pdf")
# blocks:  [{"type": "title", "page": 0, "text": "..."},
#           {"type": "authors", ...}, {"type": "paragraph", ...}, ...]
# dropped: page headers, footers and text inside figures, kept out of the flow
```

The blocks are the input of
[grain-growth-chunking](https://pypi.org/project/grain-growth-chunking/). For
the intermediate layout — regions with boxes, labels and lines, in the
`textreflow` input contract — use `layout_pdf`:

```python
from recrystal import LayoutDetector, layout_pdf

detector = LayoutDetector()                       # load once, reuse across PDFs
layout = layout_pdf("paper.pdf", detector=detector, max_pages=5)
```

`LayoutDetector(model_path=..., providers=["CPUExecutionProvider"])` takes a
local `.onnx` and an explicit provider list. On a GPU that runs out of memory,
inference drops to CPU instead of failing.

## Weights

The weights (38 MB, `.onnx`) are downloaded on first use from
[darkdwine/yolo11-doc-layout-research-papers](https://huggingface.co/darkdwine/yolo11-doc-layout-research-papers)
into `$RECRYSTAL_HOME`, default `~/.cache/recrystal`. Put the file there
yourself for offline use.

## Evaluation

From a pre-registered evaluation over 514 papers and 400 questions, round 2 of
[Anneal's report](https://github.com/mudit-choudhary/Anneal/blob/main/evals/Reports/Report.md):

- **Fewest answers lost in parsing**: 24 of 400 answer spans missing from the
  output (near match), against Docling's 48 and PyMuPDF4LLM's 61. A span the
  parser never emits cannot be retrieved by any chunker.
- **Retrieves better** under every chunker tested: pooled +0.064 on
  `span_hit_near@4000ch` against Docling and +0.122 against PyMuPDF4LLM,
  significant on 3 of 3 chunkers for each.
- **2.88× faster than Docling** per page, 0.1375 s against 0.3958, over 6,076
  pages on a 4 GB consumer GPU.

## Licence

AGPL-3.0-or-later, for two independent reasons: it links PyMuPDF (AGPL-3.0),
and the weights are fine-tuned from Ultralytics YOLO11 (AGPL-3.0). The
assembler, [textreflow](https://github.com/mudit-choudhary/textreflow), and the
chunker, [grain-growth-chunking](https://github.com/mudit-choudhary/grain-growth-chunking),
link neither and are Apache-2.0.

## Citation

See [CITATION.cff](CITATION.cff).
