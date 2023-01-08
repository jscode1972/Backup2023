
chrome.browserAction.onClicked.addListener(function(tab) {
  
  var viewTabUrl = chrome.extension.getURL('/app/portal.html');
  var imageUrl = '/images/apple.png';

  chrome.tabs.create({url : viewTabUrl});
 
});
