# metrics for generative models

Updated inception and fid scores code.

requires:

* python >= 3.6
* pytorch >= 4.0
* tensorflow >= 1.10

features:

* fixed  open-ai tensorflow inception v2 code with bs=100 for fast computation (change bs=100 -> 1 is slow, see https://github.com/openai/improved-gan/commit/0b7ed92e47ff7047701be3e10a3bd6363999f5e7)
* tests for both fid and inception v2 and v3 scores
* invoking tensorflow inception v2 score from within pytorch session

notes:

* be careful when computing inception scores, verify (1) nhwc, ncwh, rgb order, (2) normalization, (3) sample size
* see general inception score issues: https://github.com/sbarratt/inception-score-pytorch/issues?q=is%3Aissue+is%3Aclosed
* the official, widely used open-ai tensorflow v2 code appears to have a bug: https://github.com/sbarratt/inception-score-pytorch/issues/1
* read about general issues regarding the inception score: https://arxiv.org/pdf/1801.01973.pdf
* do NOT use the inception score for datasets other than imagenet (even though in the code we used it for cifar), use fid score instead
