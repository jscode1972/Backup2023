搞懂了再來重寫

### 檔案結構說明
* manifest.json
* icon.png  (呈現在網址列右邊的icon)
* popup.html (使用者點擊網址列旁的icon後，出現的POP視窗 )
* popup.js (**這一隻JS檔沒有記載在安裝檔裡**，而是在popup.html裡載入 )

### 進一步的說明什麼是Chrome Extension
* extension是一個HTML、CSS、JS、images，以及你extension中需要的任何東西，打包成的一個壓縮檔，事實上extension就是一個web pages app。
* 這個APP可以使用Broswer提供的API，諸如：Standard JavaScript APIs、XMLHttpRequest、HTML5  等，跟一般的web APP無異。
* extension可以經由 [content script](https://developer.chrome.com/extensions/content_scripts)
  或 [cross-origin XMLHttpRequests](https://developer.chrome.com/extensions/xhr) 與頁面或伺服器互動。
* extension也可以用JS與Chrome的功能互動，例如：bookmarks 及 tabs.

### Extensions 的 UI元素 
* [browser actions](https://developer.chrome.com/extensions/browserAction)：當你的extension跟大部份的網址都會有所互動的時後 。
* [page actions](https://developer.chrome.com/extensions/pageAction)： 當你的extension需要在特定的網域底下才需要啟動。
* [options page](https://developer.chrome.com/extensions/options)：用來讓使用者設定extension的參數(如果你允許的話)。
* [override page](https://developer.chrome.com/extensions/override)：用來替換點chrome新開TAB時的預設頁面、或是書籤管理頁面、或瀏覽紀錄頁(三擇一)

### extension裡有兩個種類的 Browser API
* 一種是網頁平常使用的瀏覽器API(諸如：Standard JavaScript APIs、XMLHttpRequest、HTML5  和 other emerging APIs)，
* 另一種是，Chrome獨有的，特別提供extension來調用的API。我們稱之為： chrome.* APIs

