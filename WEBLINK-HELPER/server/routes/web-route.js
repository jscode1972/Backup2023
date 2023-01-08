var express = require('express');
var router = express.Router();
var WebLinkModel = require('../models/WebLinkModel');

// middleware that is specific to this router
/************ 
router.use(function timeLog(req, res, next) {
  console.log('Time: ', Date.now());
  next();
});
************/

// POST /web/link
router.post('/link', function(req, res) {
  var query = { url: req.body.url }; 
  var update = { url: req.body.url , 
                 title: req.body.title, 
                 tags: req.body.tags,
                 keywords: req.body.keywords };
  var opts = { upsert: true }; // 找不到就新增
  //****** 法一 => 部分取代內容 (不會異動沒有指定的部分) 
  WebLinkModel.findOneAndUpdate(query, update, opts, function(err, out) { // res 代表 resolves 
    //if (err) res.send('err');
    //if (out) res.send('ok');
    res.send('done');
    res.end();
  }); 
});

// GET /web/link?url=www.xxx.com
router.get('/link', function(req, res) {
  var query = { url: req.query.url };
  //****** 法一 => 部分取代內容 (不會異動沒有指定的部分) 
  WebLinkModel.findOne(query, {}, function (err, doc) {
    if (err) {
      console.log(err);
      res.send('err');
      res.end();
    }
    console.log(doc);
    res.send(doc);
    res.end();
  }); 
});

module.exports = router;

/****************************************************************
// GET /web/root
router.get('/root', function(req, res) {
  console.log(req.query.url);
  var query = { url: { "$regex": req.query.url, "$options": "i" } };
  //****** 法一 => 部分取代內容 (不會異動沒有指定的部分) 
  WebLinkModel.findOne(query, {}, function (err, doc) {
    if (err) res.send(err);
    if (doc) res.send(doc);
    res.end();
  }); 
});

// define the list route
router.get('/list', function(req, res) {
  // 找資料
  res.send('WebLinks list page');
  //WebTagModel.find(function (err, tag) {
  //  if (err) return console.error(err);
  //  res.send(JSON.stringify(tag));
  //});
});
****************************************************************/
