from keras.datasets import reuters

# 載入 Reuters 資料集, 如果是第一次載入會自行下載資料集
top_words = 10000
(X_train, Y_train), (X_test, Y_test) = reuters.load_data(
                                       num_words=top_words)
# 最大的單字索引值
max_index = max(max(sequence) for sequence in X_train)
print("Max Index: ", max_index)
# 建立新聞的解碼字典
word_index = reuters.get_word_index()
we_index = word_index["we"]
print("'we' index:", we_index)
decode_word_map = dict([(value, key) for (key, value)
                                  in word_index.items()])
print(decode_word_map[we_index])
# 解碼顯示新聞內容
decoded_indices = [decode_word_map.get(i-3, "?")
                           for i in X_train[0]]
print(decoded_indices)
decoded_news = " ".join(decoded_indices)
print(decoded_news)