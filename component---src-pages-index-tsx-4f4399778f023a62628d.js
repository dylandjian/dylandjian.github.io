"use strict";(self.webpackChunkgatsby_starter_blog=self.webpackChunkgatsby_starter_blog||[]).push([[245],{3674:function(e,t,n){n.d(t,{z:function(){return m}});var a=n(6540),r=n(2047),s=n(8855),i=n(3720),o=n(4848),l=(0,s.R)((function(e,t){const{direction:n,align:a,justify:r,wrap:s,basis:l,grow:c,shrink:d,...u}=e,f={display:"flex",flexDirection:n,alignItems:a,justifyContent:r,flexWrap:s,flexBasis:l,flexGrow:c,flexShrink:d};return(0,o.jsx)(i.B.div,{ref:t,__css:f,...u})}));l.displayName="Flex";var c=n(4765),d=n(8539),u=n(7500),f=n(2532);function m(){const[e]=(0,r.U)("(min-width: 500px)");return a.createElement(l,{className:"bio",marginBottom:"12px"},a.createElement(c.B,{direction:e?"row":"column",alignItems:"center",spacing:8},a.createElement(l,null,a.createElement(f.S,{layout:"fixed",formats:["auto","webp","avif"],src:"../images/profile-pic.png",width:200,height:200,style:{borderRadius:"20px",minWidth:"50px"},quality:100,alt:"Profile picture",__imageData:n(838)})),a.createElement(d.T,{align:"start"},a.createElement(u.E,{fontSize:"xl",as:"b"},"Welcome to my blog !"),a.createElement(u.E,null,"I am a Software Engineer, mostly self-taught and passionated about a lot of different topics."),a.createElement(u.E,null,"On this blog, you will find blog posts about anything, recipes, photos and any other endeavors I may find interesting to share."),a.createElement(u.E,null," Happy browsing !"))))}},9892:function(e,t,n){n.r(t),n.d(t,{default:function(){return ie}});var a=n(4506),r=n(6540);var s=n(9857);function i(...e){return t=>{e.forEach((e=>{!function(e,t){if(null!=e)if("function"!=typeof e)try{e.current=t}catch(n){throw new Error(`Cannot assign value '${t}' to ref '${e}'`)}else e(t)}(e,t)}))}}function o(e){const t=e.target,{tagName:n,isContentEditable:a}=t;return"INPUT"!==n&&"TEXTAREA"!==n&&!0!==a}function l(e={}){const{ref:t,isDisabled:n,isFocusable:a,clickOnEnter:l=!0,clickOnSpace:c=!0,onMouseDown:d,onMouseUp:u,onClick:f,onKeyDown:m,onKeyUp:p,tabIndex:b,onMouseOver:h,onMouseLeave:g,...E}=e,[v,x]=(0,r.useState)(!0),[y,N]=(0,r.useState)(!1),I=function(){const e=(0,r.useRef)(new Map),t=e.current,n=(0,r.useCallback)(((t,n,a,r)=>{e.current.set(a,{type:n,el:t,options:r}),t.addEventListener(n,a,r)}),[]),a=(0,r.useCallback)(((t,n,a,r)=>{t.removeEventListener(n,a,r),e.current.delete(a)}),[]);return(0,r.useEffect)((()=>()=>{t.forEach(((e,t)=>{a(e.el,e.type,t,e.options)}))}),[a,t]),{add:n,remove:a}}(),S=v?b:b||0,w=n&&!a,k=(0,r.useCallback)((e=>{if(n)return e.stopPropagation(),void e.preventDefault();e.currentTarget.focus(),null==f||f(e)}),[n,f]),C=(0,r.useCallback)((e=>{y&&o(e)&&(e.preventDefault(),e.stopPropagation(),N(!1),I.remove(document,"keyup",C,!1))}),[y,I]),T=(0,r.useCallback)((e=>{if(null==m||m(e),n||e.defaultPrevented||e.metaKey)return;if(!o(e.nativeEvent)||v)return;const t=l&&"Enter"===e.key;if(c&&" "===e.key&&(e.preventDefault(),N(!0)),t){e.preventDefault();e.currentTarget.click()}I.add(document,"keyup",C,!1)}),[n,v,m,l,c,I,C]),_=(0,r.useCallback)((e=>{if(null==p||p(e),n||e.defaultPrevented||e.metaKey)return;if(!o(e.nativeEvent)||v)return;if(c&&" "===e.key){e.preventDefault(),N(!1);e.currentTarget.click()}}),[c,v,n,p]),D=(0,r.useCallback)((e=>{0===e.button&&(N(!1),I.remove(document,"mouseup",D,!1))}),[I]),O=(0,r.useCallback)((e=>{if(0!==e.button)return;if(n)return e.stopPropagation(),void e.preventDefault();v||N(!0);e.currentTarget.focus({preventScroll:!0}),I.add(document,"mouseup",D,!1),null==d||d(e)}),[n,v,d,I,D]),z=(0,r.useCallback)((e=>{0===e.button&&(v||N(!1),null==u||u(e))}),[u,v]),M=(0,r.useCallback)((e=>{n?e.preventDefault():null==h||h(e)}),[n,h]),B=(0,r.useCallback)((e=>{y&&(e.preventDefault(),N(!1)),null==g||g(e)}),[y,g]),j=i(t,(e=>{e&&"BUTTON"!==e.tagName&&x(!1)}));return v?{...E,ref:j,type:"button","aria-disabled":w?void 0:n,disabled:w,onClick:k,onMouseDown:d,onMouseUp:u,onKeyUp:p,onKeyDown:m,onMouseOver:h,onMouseLeave:g}:{...E,ref:j,role:"button","data-active":(0,s.sE)(y),"aria-disabled":n?"true":void 0,tabIndex:w?void 0:S,onClick:k,onMouseDown:O,onMouseUp:z,onKeyUp:_,onKeyDown:T,onMouseOver:M,onMouseLeave:B}}var c=Object.defineProperty,d=(e,t,n)=>(((e,t,n)=>{t in e?c(e,t,{enumerable:!0,configurable:!0,writable:!0,value:n}):e[t]=n})(e,"symbol"!=typeof t?t+"":t,n),n);function u(e){return e.sort(((e,t)=>{const n=e.compareDocumentPosition(t);if(n&Node.DOCUMENT_POSITION_FOLLOWING||n&Node.DOCUMENT_POSITION_CONTAINED_BY)return-1;if(n&Node.DOCUMENT_POSITION_PRECEDING||n&Node.DOCUMENT_POSITION_CONTAINS)return 1;if(n&Node.DOCUMENT_POSITION_DISCONNECTED||n&Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC)throw Error("Cannot sort the given nodes.");return 0}))}function f(e,t,n){let a=e+1;return n&&a>=t&&(a=0),a}function m(e,t,n){let a=e-1;return n&&a<0&&(a=t),a}var p="undefined"!=typeof window?r.useLayoutEffect:r.useEffect,b=e=>e,h=class{constructor(){d(this,"descendants",new Map),d(this,"register",(e=>{var t;if(null!=e)return"object"==typeof(t=e)&&"nodeType"in t&&t.nodeType===Node.ELEMENT_NODE?this.registerNode(e):t=>{this.registerNode(t,e)}})),d(this,"unregister",(e=>{this.descendants.delete(e);const t=u(Array.from(this.descendants.keys()));this.assignIndex(t)})),d(this,"destroy",(()=>{this.descendants.clear()})),d(this,"assignIndex",(e=>{this.descendants.forEach((t=>{const n=e.indexOf(t.node);t.index=n,t.node.dataset.index=t.index.toString()}))})),d(this,"count",(()=>this.descendants.size)),d(this,"enabledCount",(()=>this.enabledValues().length)),d(this,"values",(()=>Array.from(this.descendants.values()).sort(((e,t)=>e.index-t.index)))),d(this,"enabledValues",(()=>this.values().filter((e=>!e.disabled)))),d(this,"item",(e=>{if(0!==this.count())return this.values()[e]})),d(this,"enabledItem",(e=>{if(0!==this.enabledCount())return this.enabledValues()[e]})),d(this,"first",(()=>this.item(0))),d(this,"firstEnabled",(()=>this.enabledItem(0))),d(this,"last",(()=>this.item(this.descendants.size-1))),d(this,"lastEnabled",(()=>{const e=this.enabledValues().length-1;return this.enabledItem(e)})),d(this,"indexOf",(e=>{var t,n;return e&&null!=(n=null==(t=this.descendants.get(e))?void 0:t.index)?n:-1})),d(this,"enabledIndexOf",(e=>null==e?-1:this.enabledValues().findIndex((t=>t.node.isSameNode(e))))),d(this,"next",((e,t=!0)=>{const n=f(e,this.count(),t);return this.item(n)})),d(this,"nextEnabled",((e,t=!0)=>{const n=this.item(e);if(!n)return;const a=f(this.enabledIndexOf(n.node),this.enabledCount(),t);return this.enabledItem(a)})),d(this,"prev",((e,t=!0)=>{const n=m(e,this.count()-1,t);return this.item(n)})),d(this,"prevEnabled",((e,t=!0)=>{const n=this.item(e);if(!n)return;const a=m(this.enabledIndexOf(n.node),this.enabledCount()-1,t);return this.enabledItem(a)})),d(this,"registerNode",((e,t)=>{if(!e||this.descendants.has(e))return;const n=u(Array.from(this.descendants.keys()).concat(e));(null==t?void 0:t.disabled)&&(t.disabled=!!t.disabled);const a={node:e,index:-1,...t};this.descendants.set(e,a),this.assignIndex(n)}))}},g=n(61);var[E,v]=(0,g.q)({name:"DescendantsProvider",errorMessage:"useDescendantsContext must be used within DescendantsProvider"});var x=n(1295);var y=n(1117);var[N,I,S,w]=[b(E),()=>b(v()),()=>function(){const e=(0,r.useRef)(new h);return p((()=>()=>e.current.destroy())),e.current}(),e=>function(e){const t=v(),[n,a]=(0,r.useState)(-1),s=(0,r.useRef)(null);p((()=>()=>{s.current&&t.unregister(s.current)}),[]),p((()=>{if(!s.current)return;const e=Number(s.current.dataset.index);n==e||Number.isNaN(e)||a(e)}));const o=b(e?t.register(e):t.register);return{descendants:t,index:n,enabledIndex:t.enabledIndexOf(s.current),register:i(o,s)}}(e)];function k(e){var t;const{defaultIndex:n,onChange:a,index:s,isManual:i,isLazy:o,lazyBehavior:l="unmount",orientation:c="horizontal",direction:d="ltr",...u}=e,[f,m]=(0,r.useState)(null!=n?n:0),[p,b]=function(e){const{value:t,defaultValue:n,onChange:a,shouldUpdate:s=((e,t)=>e!==t)}=e,i=(0,x.c)(a),o=(0,x.c)(s),[l,c]=(0,r.useState)(n),d=void 0!==t,u=d?t:l,f=(0,x.c)((e=>{const t="function"==typeof e?e(u):e;o(u,t)&&(d||c(t),i(t))}),[d,i,u,o]);return[u,f]}({defaultValue:null!=n?n:0,value:s,onChange:a});(0,r.useEffect)((()=>{null!=s&&m(s)}),[s]);const h=S(),g=(0,r.useId)();return{id:`tabs-${null!=(t=e.id)?t:g}`,selectedIndex:p,focusedIndex:f,setSelectedIndex:b,setFocusedIndex:m,isManual:i,isLazy:o,lazyBehavior:l,orientation:c,descendants:h,direction:d,htmlProps:u}}var[C,T]=(0,g.q)({name:"TabsContext",errorMessage:"useTabsContext: `context` is undefined. Seems you forgot to wrap all tabs components within <Tabs />"});var[_,D]=(0,g.q)({});function O(e,t){return`${e}--tab-${t}`}function z(e,t){return`${e}--tabpanel-${t}`}var M=n(8855),B=n(7484),j=n(4515),P=n(3720),R=n(4848),[L,U]=(0,g.q)({name:"TabsStylesContext",errorMessage:"useTabsStyles returned is 'undefined'. Seems you forgot to wrap the components in \"<Tabs />\" "}),A=(0,M.R)((function(e,t){const n=(0,B.o5)("Tabs",e),{children:a,className:i,...o}=(0,j.MN)(e),{htmlProps:l,descendants:c,...d}=k(o),u=(0,r.useMemo)((()=>d),[d]),{isFitted:f,...m}=l,p={position:"relative",...n.root};return(0,R.jsx)(N,{value:c,children:(0,R.jsx)(C,{value:u,children:(0,R.jsx)(L,{value:n,children:(0,R.jsx)(P.B.div,{className:(0,s.cx)("chakra-tabs",i),ref:t,...m,__css:p,children:a})})})})}));A.displayName="Tabs";var F=(0,M.R)((function(e,t){const n=function(e){const{focusedIndex:t,orientation:n,direction:a}=T(),i=I(),o=(0,r.useCallback)((e=>{const r=()=>{var e;const n=i.nextEnabled(t);n&&(null==(e=n.node)||e.focus())},s=()=>{var e;const n=i.prevEnabled(t);n&&(null==(e=n.node)||e.focus())},o="horizontal"===n,l="vertical"===n,c=e.key,d="ltr"===a?"ArrowLeft":"ArrowRight",u="ltr"===a?"ArrowRight":"ArrowLeft",f={[d]:()=>o&&s(),[u]:()=>o&&r(),ArrowDown:()=>l&&r(),ArrowUp:()=>l&&s(),Home:()=>{var e;const t=i.firstEnabled();t&&(null==(e=t.node)||e.focus())},End:()=>{var e;const t=i.lastEnabled();t&&(null==(e=t.node)||e.focus())}}[c];f&&(e.preventDefault(),f(e))}),[i,t,n,a]);return{...e,role:"tablist","aria-orientation":n,onKeyDown:(0,s.Hj)(e.onKeyDown,o)}}({...e,ref:t}),a={display:"flex",...U().tablist};return(0,R.jsx)(P.B.div,{...n,className:(0,s.cx)("chakra-tabs__tablist",e.className),__css:a})}));F.displayName="TabList";var $=(0,M.R)((function(e,t){const n=U(),a=function(e){const{isDisabled:t=!1,isFocusable:n=!1,...a}=e,{setSelectedIndex:r,isManual:o,id:c,setFocusedIndex:d,selectedIndex:u}=T(),{index:f,register:m}=w({disabled:t&&!n}),p=f===u;return{...l({...a,ref:i(m,e.ref),isDisabled:t,isFocusable:n,onClick:(0,s.Hj)(e.onClick,(()=>{r(f)}))}),id:O(c,f),role:"tab",tabIndex:p?0:-1,type:"button","aria-selected":p,"aria-controls":z(c,f),onFocus:t?void 0:(0,s.Hj)(e.onFocus,(()=>{d(f),!o&&(!t||!n)&&r(f)}))}}({...e,ref:t}),r={outline:"0",display:"flex",alignItems:"center",justifyContent:"center",...n.tab};return(0,R.jsx)(P.B.button,{...a,className:(0,s.cx)("chakra-tabs__tab",e.className),__css:r})}));$.displayName="Tab";var K=(0,M.R)((function(e,t){const n=function(e){const t=T(),{id:n,selectedIndex:a}=t,s=(0,y.a)(e.children).map(((e,t)=>(0,r.createElement)(_,{key:t,value:{isSelected:t===a,id:z(n,t),tabId:O(n,t),selectedIndex:a}},e)));return{...e,children:s}}(e),a=U();return(0,R.jsx)(P.B.div,{...n,width:"100%",ref:t,className:(0,s.cx)("chakra-tabs__tab-panels",e.className),__css:a.tabpanels})}));K.displayName="TabPanels";var V=(0,M.R)((function(e,t){const n=function(e){const{children:t,...n}=e,{isLazy:a,lazyBehavior:s}=T(),{isSelected:i,id:o,tabId:l}=D(),c=(0,r.useRef)(!1);i&&(c.current=!0);const d=function(e){const{wasSelected:t,enabled:n,isSelected:a,mode:r="unmount"}=e;return!n||!!a||!("keepMounted"!==r||!t)}({wasSelected:c.current,isSelected:i,enabled:a,mode:s});return{tabIndex:0,...n,children:d?t:null,role:"tabpanel","aria-labelledby":l,hidden:!i,id:o}}({...e,ref:t}),a=U();return(0,R.jsx)(P.B.div,{outline:"0",...n,className:(0,s.cx)("chakra-tabs__tab-panel",e.className),__css:a.tabpanel})}));V.displayName="TabPanel";var G=n(3674),H=n(4399),q=n(4794),W=n(2047),Y=n(4765),J=(0,M.R)((function(e,t){const n=(0,B.Vl)("Heading",e),{className:a,...r}=(0,j.MN)(e);return(0,R.jsx)(P.B.h2,{ref:t,className:(0,s.cx)("chakra-heading",e.className),...r,__css:n})}));J.displayName="Heading";var X=n(7500),Q=n(8539),Z=n(4194),ee=n(6287),te=n(2532);const ne=e=>{let{post:t}=e;const[n]=(0,W.U)("(min-width: 500px)");return n?r.createElement(Y.B,{spacing:0},r.createElement(J,{fontSize:26,marginY:0},t.frontmatter.title),r.createElement(X.E,{fontSize:15,as:"i",marginBottom:2},`${t.frontmatter.date} - ${t.fields.readingTime.text}`),r.createElement(X.E,{fontSize:15,marginBottom:0},t.frontmatter.description||t.excerpt)):r.createElement(Y.B,{spacing:0},r.createElement(J,{fontSize:26,marginY:0},t.frontmatter.title),r.createElement(X.E,{fontSize:15,marginBottom:0},t.frontmatter.date),r.createElement(X.E,{fontSize:15,as:"i",marginBottom:0},t.fields.readingTime.text))},ae=e=>{let{posts:t}=e;return 0===t.length?r.createElement(X.E,null,"No blog posts found."):r.createElement(Q.T,{spacing:8,paddingTop:4,alignItems:"flex-start"},t.map((e=>r.createElement(q.Link,{to:e.fields.slug,style:{textDecoration:"none",color:"inherit"},key:e.fields.name},r.createElement(Z.z,{spacing:4},r.createElement(ee.az,{minWidth:120,height:120},r.createElement(te.G,{objectFit:"cover",image:(0,te.c)(e.icon.childImageSharp),alt:"Caffe Latte"})),r.createElement(Q.T,{spacing:0,align:"flex-start"},r.createElement(ne,{post:e})))))))},re=e=>{let{post:t}=e;const[n]=(0,W.U)("(min-width: 500px)");return n?r.createElement(Y.B,{spacing:0},r.createElement(J,{fontSize:26,marginY:0},t.frontmatter.title),r.createElement(X.E,{fontSize:15,as:"i",marginBottom:2},`${t.frontmatter.date}`),r.createElement(X.E,{fontSize:15,marginBottom:0},t.frontmatter.description||t.excerpt)):r.createElement(Y.B,{spacing:0},r.createElement(J,{fontSize:26,marginY:0},t.frontmatter.title),r.createElement(X.E,{fontSize:15,marginBottom:0},t.frontmatter.date))},se=e=>{let{images:t}=e;return 0===t.length?r.createElement(X.E,null,"No blog images found."):r.createElement(Q.T,{spacing:8,paddingTop:4,alignItems:"flex-start"},t.map((e=>r.createElement(q.Link,{to:e.fields.slug,style:{textDecoration:"none",color:"inherit"},key:e.fields.name},r.createElement(Z.z,{spacing:4},r.createElement(ee.az,{minWidth:120,height:120},r.createElement(te.G,{objectFit:"cover",image:(0,te.c)(e.icon.childImageSharp),alt:"Caffe Latte"})),r.createElement(Q.T,{spacing:0,align:"flex-start"},r.createElement(re,{post:e})))))))};var ie=e=>{var t;let{data:n,location:s}=e;const i=(null===(t=n.site.siteMetadata)||void 0===t?void 0:t.title)||"Title",o=n.allMarkdownRemark.nodes.filter((e=>"blog"===e.fields.type)).reduce(((e,t)=>{const r=t.fields.name,s=n.allFile.nodes.find((e=>e.childImageSharp&&e.childImageSharp.fixed.originalName===`${r}-icon.png`));return[].concat((0,a.A)(e),[{...t,icon:s}])}),[]),l=n.allMarkdownRemark.nodes.filter((e=>"gallery"===e.fields.type)).reduce(((e,t)=>{const r=n.allFile.nodes,s=t.fields.name,i=r.find((e=>e.childImageSharp&&e.childImageSharp.fixed.originalName===`${s}-icon.png`)),o=r.filter((e=>"md"!==e.extension&&e.relativeDirectory===s));return[].concat((0,a.A)(e),[{...t,images:o,icon:i}])}),[]);return r.createElement(H.P,{location:s,title:i},r.createElement(G.z,null),r.createElement(A,{isFitted:!0,defaultIndex:0},r.createElement(F,null,r.createElement($,null,"Posts"),r.createElement($,null,"Gallery"),r.createElement($,null,"Recipes")),r.createElement(K,null,r.createElement(V,null,r.createElement(ae,{posts:o})),r.createElement(V,null,r.createElement(se,{images:l})),r.createElement(V,null,"No recipes yet"))))}},8539:function(e,t,n){n.d(t,{T:function(){return i}});var a=n(4765),r=n(8855),s=n(4848),i=(0,r.R)(((e,t)=>(0,s.jsx)(a.B,{align:"center",...e,direction:"column",ref:t})));i.displayName="VStack"},6287:function(e,t,n){n.d(t,{az:function(){return i}});var a=n(3720),r=n(8855),s=n(4848),i=(0,a.B)("div");i.displayName="Box";var o=(0,r.R)((function(e,t){const{size:n,centerContent:a=!0,...r}=e,o=a?{display:"flex",alignItems:"center",justifyContent:"center"}:{};return(0,s.jsx)(i,{ref:t,boxSize:n,__css:{...o,flexShrink:0,flexGrow:0},...r})}));o.displayName="Square",(0,r.R)((function(e,t){const{size:n,...a}=e;return(0,s.jsx)(o,{size:n,ref:t,borderRadius:"9999px",...a})})).displayName="Circle"},838:function(e){e.exports=JSON.parse('{"layout":"fixed","backgroundColor":"#d8d8d8","images":{"fallback":{"src":"/static/6ef968cfebb62e6eda1d4c39e3772686/839ae/profile-pic.jpg","srcSet":"/static/6ef968cfebb62e6eda1d4c39e3772686/839ae/profile-pic.jpg 200w,\\n/static/6ef968cfebb62e6eda1d4c39e3772686/10998/profile-pic.jpg 400w","sizes":"200px"},"sources":[{"srcSet":"/static/6ef968cfebb62e6eda1d4c39e3772686/73fca/profile-pic.avif 200w,\\n/static/6ef968cfebb62e6eda1d4c39e3772686/44a11/profile-pic.avif 400w","type":"image/avif","sizes":"200px"},{"srcSet":"/static/6ef968cfebb62e6eda1d4c39e3772686/8b00d/profile-pic.webp 200w,\\n/static/6ef968cfebb62e6eda1d4c39e3772686/9c0a1/profile-pic.webp 400w","type":"image/webp","sizes":"200px"}]},"width":200,"height":200}')}}]);
//# sourceMappingURL=component---src-pages-index-tsx-4f4399778f023a62628d.js.map