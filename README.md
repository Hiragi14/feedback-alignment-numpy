# feedback-alignment-numpy
BackpropagationとFeedback Alignmentの精度評価を三層MLPを用いて行う。
使用したデータセットはMNISTで、使用言語はPythonである。

## 使い方
`train.py`の実行で学習が始まる。学習の精度と損失は、`data`フォルダに保存される。
`graph.py`を実行することで、精度と損失のグラフが.pngファイルとして`image`フォルダに書き出される。