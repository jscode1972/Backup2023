document.addEventListener('DOMContentLoaded', function() {
    var btn = document.getElementById('btn');
    // onClick's logic below:
    btn.addEventListener('click', function() {
        hello('fuck you');
    });
});

function hello(aa) {
   var demo = document.getElementById('demo');
   demo.innerText = aa;
}




