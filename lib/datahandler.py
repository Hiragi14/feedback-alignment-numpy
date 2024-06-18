import numpy as np

# バッチ作成
def create_batch(data, batch_size):
    # 商と余りを取得
    num_batches, mod = divmod(data.shape[0], batch_size)
    # modを除いたデータをsplit
    batched_data = np.split(data[: batch_size * num_batches], num_batches)
    # modがあれば最後に追加する
    if mod:
        batched_data.append(data[batch_size * num_batches:])
    return batched_data