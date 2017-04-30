<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>

<title>Introduction</title>

<script type="text/javascript">
window.onload = function() {
  var imgs = document.getElementsByTagName('img'), i, img;
  for (i = 0; i < imgs.length; i++) {
    img = imgs[i];
    // center an image if it is the only element of its parent
    if (img.parentElement.childElementCount === 1)
      img.parentElement.style.textAlign = 'center';
  }
};
</script>

<!-- Styles for R syntax highlighter -->
<style type="text/css">
   pre .operator,
   pre .paren {
     color: rgb(104, 118, 135)
   }

   pre .literal {
     color: #990073
   }

   pre .number {
     color: #099;
   }

   pre .comment {
     color: #998;
     font-style: italic
   }

   pre .keyword {
     color: #900;
     font-weight: bold
   }

   pre .identifier {
     color: rgb(0, 0, 0);
   }

   pre .string {
     color: #d14;
   }
</style>

<!-- R syntax highlighter -->
<script type="text/javascript">
var hljs=new function(){function m(p){return p.replace(/&/gm,"&amp;").replace(/</gm,"&lt;")}function f(r,q,p){return RegExp(q,"m"+(r.cI?"i":"")+(p?"g":""))}function b(r){for(var p=0;p<r.childNodes.length;p++){var q=r.childNodes[p];if(q.nodeName=="CODE"){return q}if(!(q.nodeType==3&&q.nodeValue.match(/\s+/))){break}}}function h(t,s){var p="";for(var r=0;r<t.childNodes.length;r++){if(t.childNodes[r].nodeType==3){var q=t.childNodes[r].nodeValue;if(s){q=q.replace(/\n/g,"")}p+=q}else{if(t.childNodes[r].nodeName=="BR"){p+="\n"}else{p+=h(t.childNodes[r])}}}if(/MSIE [678]/.test(navigator.userAgent)){p=p.replace(/\r/g,"\n")}return p}function a(s){var r=s.className.split(/\s+/);r=r.concat(s.parentNode.className.split(/\s+/));for(var q=0;q<r.length;q++){var p=r[q].replace(/^language-/,"");if(e[p]){return p}}}function c(q){var p=[];(function(s,t){for(var r=0;r<s.childNodes.length;r++){if(s.childNodes[r].nodeType==3){t+=s.childNodes[r].nodeValue.length}else{if(s.childNodes[r].nodeName=="BR"){t+=1}else{if(s.childNodes[r].nodeType==1){p.push({event:"start",offset:t,node:s.childNodes[r]});t=arguments.callee(s.childNodes[r],t);p.push({event:"stop",offset:t,node:s.childNodes[r]})}}}}return t})(q,0);return p}function k(y,w,x){var q=0;var z="";var s=[];function u(){if(y.length&&w.length){if(y[0].offset!=w[0].offset){return(y[0].offset<w[0].offset)?y:w}else{return w[0].event=="start"?y:w}}else{return y.length?y:w}}function t(D){var A="<"+D.nodeName.toLowerCase();for(var B=0;B<D.attributes.length;B++){var C=D.attributes[B];A+=" "+C.nodeName.toLowerCase();if(C.value!==undefined&&C.value!==false&&C.value!==null){A+='="'+m(C.value)+'"'}}return A+">"}while(y.length||w.length){var v=u().splice(0,1)[0];z+=m(x.substr(q,v.offset-q));q=v.offset;if(v.event=="start"){z+=t(v.node);s.push(v.node)}else{if(v.event=="stop"){var p,r=s.length;do{r--;p=s[r];z+=("</"+p.nodeName.toLowerCase()+">")}while(p!=v.node);s.splice(r,1);while(r<s.length){z+=t(s[r]);r++}}}}return z+m(x.substr(q))}function j(){function q(x,y,v){if(x.compiled){return}var u;var s=[];if(x.k){x.lR=f(y,x.l||hljs.IR,true);for(var w in x.k){if(!x.k.hasOwnProperty(w)){continue}if(x.k[w] instanceof Object){u=x.k[w]}else{u=x.k;w="keyword"}for(var r in u){if(!u.hasOwnProperty(r)){continue}x.k[r]=[w,u[r]];s.push(r)}}}if(!v){if(x.bWK){x.b="\\b("+s.join("|")+")\\s"}x.bR=f(y,x.b?x.b:"\\B|\\b");if(!x.e&&!x.eW){x.e="\\B|\\b"}if(x.e){x.eR=f(y,x.e)}}if(x.i){x.iR=f(y,x.i)}if(x.r===undefined){x.r=1}if(!x.c){x.c=[]}x.compiled=true;for(var t=0;t<x.c.length;t++){if(x.c[t]=="self"){x.c[t]=x}q(x.c[t],y,false)}if(x.starts){q(x.starts,y,false)}}for(var p in e){if(!e.hasOwnProperty(p)){continue}q(e[p].dM,e[p],true)}}function d(B,C){if(!j.called){j();j.called=true}function q(r,M){for(var L=0;L<M.c.length;L++){if((M.c[L].bR.exec(r)||[null])[0]==r){return M.c[L]}}}function v(L,r){if(D[L].e&&D[L].eR.test(r)){return 1}if(D[L].eW){var M=v(L-1,r);return M?M+1:0}return 0}function w(r,L){return L.i&&L.iR.test(r)}function K(N,O){var M=[];for(var L=0;L<N.c.length;L++){M.push(N.c[L].b)}var r=D.length-1;do{if(D[r].e){M.push(D[r].e)}r--}while(D[r+1].eW);if(N.i){M.push(N.i)}return f(O,M.join("|"),true)}function p(M,L){var N=D[D.length-1];if(!N.t){N.t=K(N,E)}N.t.lastIndex=L;var r=N.t.exec(M);return r?[M.substr(L,r.index-L),r[0],false]:[M.substr(L),"",true]}function z(N,r){var L=E.cI?r[0].toLowerCase():r[0];var M=N.k[L];if(M&&M instanceof Array){return M}return false}function F(L,P){L=m(L);if(!P.k){return L}var r="";var O=0;P.lR.lastIndex=0;var M=P.lR.exec(L);while(M){r+=L.substr(O,M.index-O);var N=z(P,M);if(N){x+=N[1];r+='<span class="'+N[0]+'">'+M[0]+"</span>"}else{r+=M[0]}O=P.lR.lastIndex;M=P.lR.exec(L)}return r+L.substr(O,L.length-O)}function J(L,M){if(M.sL&&e[M.sL]){var r=d(M.sL,L);x+=r.keyword_count;return r.value}else{return F(L,M)}}function I(M,r){var L=M.cN?'<span class="'+M.cN+'">':"";if(M.rB){y+=L;M.buffer=""}else{if(M.eB){y+=m(r)+L;M.buffer=""}else{y+=L;M.buffer=r}}D.push(M);A+=M.r}function G(N,M,Q){var R=D[D.length-1];if(Q){y+=J(R.buffer+N,R);return false}var P=q(M,R);if(P){y+=J(R.buffer+N,R);I(P,M);return P.rB}var L=v(D.length-1,M);if(L){var O=R.cN?"</span>":"";if(R.rE){y+=J(R.buffer+N,R)+O}else{if(R.eE){y+=J(R.buffer+N,R)+O+m(M)}else{y+=J(R.buffer+N+M,R)+O}}while(L>1){O=D[D.length-2].cN?"</span>":"";y+=O;L--;D.length--}var r=D[D.length-1];D.length--;D[D.length-1].buffer="";if(r.starts){I(r.starts,"")}return R.rE}if(w(M,R)){throw"Illegal"}}var E=e[B];var D=[E.dM];var A=0;var x=0;var y="";try{var s,u=0;E.dM.buffer="";do{s=p(C,u);var t=G(s[0],s[1],s[2]);u+=s[0].length;if(!t){u+=s[1].length}}while(!s[2]);if(D.length>1){throw"Illegal"}return{r:A,keyword_count:x,value:y}}catch(H){if(H=="Illegal"){return{r:0,keyword_count:0,value:m(C)}}else{throw H}}}function g(t){var p={keyword_count:0,r:0,value:m(t)};var r=p;for(var q in e){if(!e.hasOwnProperty(q)){continue}var s=d(q,t);s.language=q;if(s.keyword_count+s.r>r.keyword_count+r.r){r=s}if(s.keyword_count+s.r>p.keyword_count+p.r){r=p;p=s}}if(r.language){p.second_best=r}return p}function i(r,q,p){if(q){r=r.replace(/^((<[^>]+>|\t)+)/gm,function(t,w,v,u){return w.replace(/\t/g,q)})}if(p){r=r.replace(/\n/g,"<br>")}return r}function n(t,w,r){var x=h(t,r);var v=a(t);var y,s;if(v){y=d(v,x)}else{return}var q=c(t);if(q.length){s=document.createElement("pre");s.innerHTML=y.value;y.value=k(q,c(s),x)}y.value=i(y.value,w,r);var u=t.className;if(!u.match("(\\s|^)(language-)?"+v+"(\\s|$)")){u=u?(u+" "+v):v}if(/MSIE [678]/.test(navigator.userAgent)&&t.tagName=="CODE"&&t.parentNode.tagName=="PRE"){s=t.parentNode;var p=document.createElement("div");p.innerHTML="<pre><code>"+y.value+"</code></pre>";t=p.firstChild.firstChild;p.firstChild.cN=s.cN;s.parentNode.replaceChild(p.firstChild,s)}else{t.innerHTML=y.value}t.className=u;t.result={language:v,kw:y.keyword_count,re:y.r};if(y.second_best){t.second_best={language:y.second_best.language,kw:y.second_best.keyword_count,re:y.second_best.r}}}function o(){if(o.called){return}o.called=true;var r=document.getElementsByTagName("pre");for(var p=0;p<r.length;p++){var q=b(r[p]);if(q){n(q,hljs.tabReplace)}}}function l(){if(window.addEventListener){window.addEventListener("DOMContentLoaded",o,false);window.addEventListener("load",o,false)}else{if(window.attachEvent){window.attachEvent("onload",o)}else{window.onload=o}}}var e={};this.LANGUAGES=e;this.highlight=d;this.highlightAuto=g;this.fixMarkup=i;this.highlightBlock=n;this.initHighlighting=o;this.initHighlightingOnLoad=l;this.IR="[a-zA-Z][a-zA-Z0-9_]*";this.UIR="[a-zA-Z_][a-zA-Z0-9_]*";this.NR="\\b\\d+(\\.\\d+)?";this.CNR="\\b(0[xX][a-fA-F0-9]+|(\\d+(\\.\\d*)?|\\.\\d+)([eE][-+]?\\d+)?)";this.BNR="\\b(0b[01]+)";this.RSR="!|!=|!==|%|%=|&|&&|&=|\\*|\\*=|\\+|\\+=|,|\\.|-|-=|/|/=|:|;|<|<<|<<=|<=|=|==|===|>|>=|>>|>>=|>>>|>>>=|\\?|\\[|\\{|\\(|\\^|\\^=|\\||\\|=|\\|\\||~";this.ER="(?![\\s\\S])";this.BE={b:"\\\\.",r:0};this.ASM={cN:"string",b:"'",e:"'",i:"\\n",c:[this.BE],r:0};this.QSM={cN:"string",b:'"',e:'"',i:"\\n",c:[this.BE],r:0};this.CLCM={cN:"comment",b:"//",e:"$"};this.CBLCLM={cN:"comment",b:"/\\*",e:"\\*/"};this.HCM={cN:"comment",b:"#",e:"$"};this.NM={cN:"number",b:this.NR,r:0};this.CNM={cN:"number",b:this.CNR,r:0};this.BNM={cN:"number",b:this.BNR,r:0};this.inherit=function(r,s){var p={};for(var q in r){p[q]=r[q]}if(s){for(var q in s){p[q]=s[q]}}return p}}();hljs.LANGUAGES.cpp=function(){var a={keyword:{"false":1,"int":1,"float":1,"while":1,"private":1,"char":1,"catch":1,"export":1,virtual:1,operator:2,sizeof:2,dynamic_cast:2,typedef:2,const_cast:2,"const":1,struct:1,"for":1,static_cast:2,union:1,namespace:1,unsigned:1,"long":1,"throw":1,"volatile":2,"static":1,"protected":1,bool:1,template:1,mutable:1,"if":1,"public":1,friend:2,"do":1,"return":1,"goto":1,auto:1,"void":2,"enum":1,"else":1,"break":1,"new":1,extern:1,using:1,"true":1,"class":1,asm:1,"case":1,typeid:1,"short":1,reinterpret_cast:2,"default":1,"double":1,register:1,explicit:1,signed:1,typename:1,"try":1,"this":1,"switch":1,"continue":1,wchar_t:1,inline:1,"delete":1,alignof:1,char16_t:1,char32_t:1,constexpr:1,decltype:1,noexcept:1,nullptr:1,static_assert:1,thread_local:1,restrict:1,_Bool:1,complex:1},built_in:{std:1,string:1,cin:1,cout:1,cerr:1,clog:1,stringstream:1,istringstream:1,ostringstream:1,auto_ptr:1,deque:1,list:1,queue:1,stack:1,vector:1,map:1,set:1,bitset:1,multiset:1,multimap:1,unordered_set:1,unordered_map:1,unordered_multiset:1,unordered_multimap:1,array:1,shared_ptr:1}};return{dM:{k:a,i:"</",c:[hljs.CLCM,hljs.CBLCLM,hljs.QSM,{cN:"string",b:"'\\\\?.",e:"'",i:"."},{cN:"number",b:"\\b(\\d+(\\.\\d*)?|\\.\\d+)(u|U|l|L|ul|UL|f|F)"},hljs.CNM,{cN:"preprocessor",b:"#",e:"$"},{cN:"stl_container",b:"\\b(deque|list|queue|stack|vector|map|set|bitset|multiset|multimap|unordered_map|unordered_set|unordered_multiset|unordered_multimap|array)\\s*<",e:">",k:a,r:10,c:["self"]}]}}}();hljs.LANGUAGES.r={dM:{c:[hljs.HCM,{cN:"number",b:"\\b0[xX][0-9a-fA-F]+[Li]?\\b",e:hljs.IMMEDIATE_RE,r:0},{cN:"number",b:"\\b\\d+(?:[eE][+\\-]?\\d*)?L\\b",e:hljs.IMMEDIATE_RE,r:0},{cN:"number",b:"\\b\\d+\\.(?!\\d)(?:i\\b)?",e:hljs.IMMEDIATE_RE,r:1},{cN:"number",b:"\\b\\d+(?:\\.\\d*)?(?:[eE][+\\-]?\\d*)?i?\\b",e:hljs.IMMEDIATE_RE,r:0},{cN:"number",b:"\\.\\d+(?:[eE][+\\-]?\\d*)?i?\\b",e:hljs.IMMEDIATE_RE,r:1},{cN:"keyword",b:"(?:tryCatch|library|setGeneric|setGroupGeneric)\\b",e:hljs.IMMEDIATE_RE,r:10},{cN:"keyword",b:"\\.\\.\\.",e:hljs.IMMEDIATE_RE,r:10},{cN:"keyword",b:"\\.\\.\\d+(?![\\w.])",e:hljs.IMMEDIATE_RE,r:10},{cN:"keyword",b:"\\b(?:function)",e:hljs.IMMEDIATE_RE,r:2},{cN:"keyword",b:"(?:if|in|break|next|repeat|else|for|return|switch|while|try|stop|warning|require|attach|detach|source|setMethod|setClass)\\b",e:hljs.IMMEDIATE_RE,r:1},{cN:"literal",b:"(?:NA|NA_integer_|NA_real_|NA_character_|NA_complex_)\\b",e:hljs.IMMEDIATE_RE,r:10},{cN:"literal",b:"(?:NULL|TRUE|FALSE|T|F|Inf|NaN)\\b",e:hljs.IMMEDIATE_RE,r:1},{cN:"identifier",b:"[a-zA-Z.][a-zA-Z0-9._]*\\b",e:hljs.IMMEDIATE_RE,r:0},{cN:"operator",b:"<\\-(?!\\s*\\d)",e:hljs.IMMEDIATE_RE,r:2},{cN:"operator",b:"\\->|<\\-",e:hljs.IMMEDIATE_RE,r:1},{cN:"operator",b:"%%|~",e:hljs.IMMEDIATE_RE},{cN:"operator",b:">=|<=|==|!=|\\|\\||&&|=|\\+|\\-|\\*|/|\\^|>|<|!|&|\\||\\$|:",e:hljs.IMMEDIATE_RE,r:0},{cN:"operator",b:"%",e:"%",i:"\\n",r:1},{cN:"identifier",b:"`",e:"`",r:0},{cN:"string",b:'"',e:'"',c:[hljs.BE],r:0},{cN:"string",b:"'",e:"'",c:[hljs.BE],r:0},{cN:"paren",b:"[[({\\])}]",e:hljs.IMMEDIATE_RE,r:0}]}};
hljs.initHighlightingOnLoad();
</script>



<style type="text/css">
body, td {
   font-family: sans-serif;
   background-color: white;
   font-size: 13px;
}

body {
  max-width: 800px;
  margin: auto;
  padding: 1em;
  line-height: 20px;
}

tt, code, pre {
   font-family: 'DejaVu Sans Mono', 'Droid Sans Mono', 'Lucida Console', Consolas, Monaco, monospace;
}

h1 {
   font-size:2.2em;
}

h2 {
   font-size:1.8em;
}

h3 {
   font-size:1.4em;
}

h4 {
   font-size:1.0em;
}

h5 {
   font-size:0.9em;
}

h6 {
   font-size:0.8em;
}

a:visited {
   color: rgb(50%, 0%, 50%);
}

pre, img {
  max-width: 100%;
}
pre {
  overflow-x: auto;
}
pre code {
   display: block; padding: 0.5em;
}

code {
  font-size: 92%;
  border: 1px solid #ccc;
}

code[class] {
  background-color: #F8F8F8;
}

table, td, th {
  border: none;
}

blockquote {
   color:#666666;
   margin:0;
   padding-left: 1em;
   border-left: 0.5em #EEE solid;
}

hr {
   height: 0px;
   border-bottom: none;
   border-top-width: thin;
   border-top-style: dotted;
   border-top-color: #999999;
}

@media print {
   * {
      background: transparent !important;
      color: black !important;
      filter:none !important;
      -ms-filter: none !important;
   }

   body {
      font-size:12pt;
      max-width:100%;
   }

   a, a:visited {
      text-decoration: underline;
   }

   hr {
      visibility: hidden;
      page-break-before: always;
   }

   pre, blockquote {
      padding-right: 1em;
      page-break-inside: avoid;
   }

   tr, img {
      page-break-inside: avoid;
   }

   img {
      max-width: 100% !important;
   }

   @page :left {
      margin: 15mm 20mm 15mm 10mm;
   }

   @page :right {
      margin: 15mm 10mm 15mm 20mm;
   }

   p, h2, h3 {
      orphans: 3; widows: 3;
   }

   h2, h3 {
      page-break-after: avoid;
   }
}
</style>



</head>

<body>
<h1>Introduction</h1>

<p>Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.</p>

<p>In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.</p>

<h1>Getting data</h1>

<pre><code class="r">setwd(&quot;F:/ANALYTICS DATA/R/DATA MANIPULATION/practical machine learning&quot;)

library(readr)

train &lt;- read.csv(&quot;pml-training.csv&quot;)
test &lt;- read.csv(&quot;pml-testing.csv&quot;)

table(train$classe)
</code></pre>

<pre><code>## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
</code></pre>

<pre><code class="r">table(train$user_name)
</code></pre>

<pre><code>## 
##   adelmo carlitos  charles   eurico   jeremy    pedro 
##     3892     3112     3536     3070     3402     2610
</code></pre>

<pre><code class="r">table(train$user_name,train$classe)
</code></pre>

<pre><code>##           
##               A    B    C    D    E
##   adelmo   1165  776  750  515  686
##   carlitos  834  690  493  486  609
##   charles   899  745  539  642  711
##   eurico    865  592  489  582  542
##   jeremy   1177  489  652  522  562
##   pedro     640  505  499  469  497
</code></pre>

<pre><code class="r">prop.table(table(train$user_name,train$classe),1)
</code></pre>

<pre><code>##           
##                    A         B         C         D         E
##   adelmo   0.2993320 0.1993834 0.1927030 0.1323227 0.1762590
##   carlitos 0.2679949 0.2217224 0.1584190 0.1561697 0.1956941
##   charles  0.2542421 0.2106900 0.1524321 0.1815611 0.2010747
##   eurico   0.2817590 0.1928339 0.1592834 0.1895765 0.1765472
##   jeremy   0.3459730 0.1437390 0.1916520 0.1534392 0.1651969
##   pedro    0.2452107 0.1934866 0.1911877 0.1796935 0.1904215
</code></pre>

<h1>Cleaning data</h1>

<pre><code class="r">#Doing some basic  some basic data clean-up by removing columns 1 to 6, 
#which are there just for information and reference purposes:

train &lt;- train[,7:160]
test &lt;- test[,7:160]

# and removing all columns that mostly contain NA&#39;s

is_data  &lt;- apply(!is.na(train), 2, sum) &gt; 19621  # which is the number of observations
train &lt;- train[, is_data]
test &lt;- test[, is_data]
</code></pre>

<h1>Data Partitioning</h1>

<pre><code class="r">#Before we can move forward with data analysis, we split the training set into
#two for cross validation purposes. We randomly
#subsample 60% of the set for
#training purposes (actual model building), while the 40% remainder will be used 
#only for testing, evaluation and accuracy measurement.



library(caret)
</code></pre>

<pre><code>## Loading required package: lattice
</code></pre>

<pre><code>## Loading required package: ggplot2
</code></pre>

<pre><code class="r">set.seed(3141592)
inTrain &lt;- createDataPartition(y=train$classe, p=0.60, list=FALSE)
train1  &lt;- train[inTrain,]
train2  &lt;- train[-inTrain,]
dim(train1)
</code></pre>

<pre><code>## [1] 11776    87
</code></pre>

<h1>Removing non-zero variables from the dataset</h1>

<pre><code class="r">nzv_cols &lt;- nearZeroVar(train1)
if(length(nzv_cols) &gt; 0) {
train1 &lt;- train1[, -nzv_cols]
train2 &lt;- train2[, -nzv_cols]
}
dim(train1)
</code></pre>

<pre><code>## [1] 11776    54
</code></pre>

<h1>Data Manipulation</h1>

<pre><code class="r">#lets look at the relative importance of the variables

library(randomForest)
</code></pre>

<pre><code>## randomForest 4.6-12
</code></pre>

<pre><code>## Type rfNews() to see new features/changes/bug fixes.
</code></pre>

<pre><code>## 
## Attaching package: &#39;randomForest&#39;
</code></pre>

<pre><code>## The following object is masked from &#39;package:ggplot2&#39;:
## 
##     margin
</code></pre>

<pre><code class="r">set.seed(3141592)
fitModel &lt;- randomForest(classe~., data=train1, importance=TRUE, ntree=100)
varImpPlot(fitModel)
</code></pre>

<p><img src="data:image/png;base64,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" alt="plot of chunk unnamed-chunk-5"/></p>

<pre><code class="r">#Using the Accuracy and Gini graphs above, we select the top 10 variables that we&#39;ll use for model
#building. If the accuracy of the resulting model is acceptable, limiting the number of variables 
#is a good idea to ensure readability and interpretability of the model. 

#A model with 10 parameters is certainly much more user friendly than a model with 53 parameters.

#Our 10 covariates are: yaw_belt, roll_belt, num_window, pitch_belt, magnet_dumbbell_y, 
#magnet_dumbbell_z, pitch_forearm, accel_dumbbell_y, roll_arm, and roll_forearm.

#Let&#39;s analyze the correlations between these 10 variables. The following code calculates 
#the correlation matrix, replaces the 1s in the diagonal with 0s, and outputs which variables
#have an absolute value correlation above 75%:
</code></pre>

<h1>Finding Correlation between variables</h1>

<pre><code class="r">correl = cor(train1[,c(&quot;yaw_belt&quot;,&quot;roll_belt&quot;,&quot;num_window&quot;,&quot;pitch_belt&quot;,&quot;magnet_dumbbell_z&quot;,&quot;magnet_dumbbell_y&quot;,&quot;pitch_forearm&quot;,&quot;accel_dumbbell_y&quot;,&quot;roll_arm&quot;,&quot;roll_forearm&quot;)])
diag(correl) &lt;- 0
which(abs(correl)&gt;0.75, arr.ind=TRUE)
</code></pre>

<pre><code>##           row col
## roll_belt   2   1
## yaw_belt    1   2
</code></pre>

<pre><code class="r">#So we may have a problem with roll_belt and yaw_belt which have a high correlation (above 75%) with each other:

cor(train1$roll_belt, train1$yaw_belt)
</code></pre>

<pre><code>## [1] 0.8152349
</code></pre>

<p>We can identify an interesting relationship between roll_belt and magnet_dumbbell_y:</p>

<pre><code class="r">qplot(roll_belt, magnet_dumbbell_y, colour=classe, data=train1)
</code></pre>

<p><img src="data:image/png;base64,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" alt="plot of chunk unnamed-chunk-7"/></p>

<pre><code class="r"># Incidentally, a quick tree classifier selects roll_belt as the first discriminant among all 53 covariates (which explains why we have eliminated 
#yaw_belt instead of roll_belt, and not the opposite: it is a &quot;more important&quot; covariate):

library(rpart.plot)
</code></pre>

<pre><code>## Loading required package: rpart
</code></pre>

<pre><code class="r">fitModel &lt;- rpart(classe~., data=train1, method=&quot;class&quot;)
prp(fitModel)
</code></pre>

<p><img src="data:image/png;base64,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" alt="plot of chunk unnamed-chunk-7"/></p>

<h1>Data Modeling</h1>

<pre><code class="r">##We are now ready to create our model.
##We are using a Random Forest algorithm, using the train() function from the caret package.
##We are using 9 variables out of the 53 as model parameters. These variables were among the 
##most significant variables generated by an initial 
##Random Forest algorithm, and are roll_belt, num_window, pitch_belt, magnet_dumbbell_y, 
##magnet_dumbbell_z, pitch_forearm, accel_dumbbell_y, roll_arm, and roll_forearm. 
##These variable are relatively independent as the maximum correlation among them is 50.57%.
##We are using a 2-fold cross-validation control. This is the simplest k-fold cross-validation ##possible and it will give a reduced computation time. 
##Because the data set is large, using a small number of folds is justified.

set.seed(3141592)
fitModel &lt;- train(classe~roll_belt+num_window+pitch_belt+magnet_dumbbell_y+magnet_dumbbell_z+pitch_forearm+accel_dumbbell_y+roll_arm+roll_forearm,
                  data=train1,
                  method=&quot;rf&quot;,
                  trControl=trainControl(method=&quot;cv&quot;,number=2),
                  prox=TRUE,
                  verbose=TRUE,
                  allowParallel=TRUE)

saveRDS(fitModel, &quot;modelRF.Rds&quot;)

##We can later use this tree, by allocating it directly to a variable using the command:

fitModel &lt;- readRDS(&quot;modelRF.Rds&quot;)


predictions &lt;- predict(fitModel, newdata=train2)
confusionMat &lt;- confusionMatrix(predictions, train2$classe)
confusionMat
</code></pre>

<pre><code>## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    1    0    0    0
##          B    1 1513    0    0    1
##          C    0    4 1368    6    1
##          D    0    0    0 1280    3
##          E    0    0    0    0 1437
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9978          
##                  95% CI : (0.9965, 0.9987)
##     No Information Rate : 0.2845          
##     P-Value [Acc &gt; NIR] : &lt; 2.2e-16       
##                                           
##                   Kappa : 0.9973          
##  Mcnemar&#39;s Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9967   1.0000   0.9953   0.9965
## Specificity            0.9998   0.9997   0.9983   0.9995   1.0000
## Pos Pred Value         0.9996   0.9987   0.9920   0.9977   1.0000
## Neg Pred Value         0.9998   0.9992   1.0000   0.9991   0.9992
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1928   0.1744   0.1631   0.1832
## Detection Prevalence   0.2845   0.1931   0.1758   0.1635   0.1832
## Balanced Accuracy      0.9997   0.9982   0.9992   0.9974   0.9983
</code></pre>

<p>Accuracy is 99.78%,which is quite good.</p>

<h2>Error Rate</h2>

<pre><code class="r">missClass = function(values, predicted) {
  sum(predicted != values) / length(values)
}
OOS_errRate = missClass(train2$classe, predictions)
OOS_errRate
</code></pre>

<pre><code>## [1] 0.002166709
</code></pre>

<p>The out-of-sample error rate is 0.23%.</p>

<h1>Making Predictions from test dataset</h1>

<pre><code class="r">predictions &lt;- predict(fitModel, newdata=test)
test$classe &lt;- predictions

submit &lt;- data.frame(problem_id = test$problem_id, classe = predictions)
write.csv(submit, file = &quot;coursera-submission.csv&quot;, row.names = FALSE)
</code></pre>

<p>#Conclusion</p>

<p>In this assignment, we accurately predicted the classification of 20 observations using a Random Forest algorithm trained on a subset of data using less than 20% of the covariates.</p>

<p>The accuracy obtained (accuracy = 99.77%, and out-of-sample error = 0.23%)</p>



</body>

</html>
