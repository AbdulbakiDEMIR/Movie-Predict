import {config, pageQueryParameterKey } from "../config.js"
import { renderApp } from "../routing/routing.js";




window.changePage = function (newPage) {
    if(newPage===""){
        history.pushState(null, "", config.basePath);
    }
    else{
        history.pushState(null, "", config.basePath+"?"+pageQueryParameterKey+"="+newPage); // URL'yi değiştir
    }
    renderApp(); 
}

window.scrollToSection = function (event) { 
    event.preventDefault();
    const href = event.target.getAttribute("href");
    const sectionId = href.substring(1);
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: "smooth" });
    } 
};
window.changeTitle = function (title){
    document.title = title;
}


export const spinner = function(isSpinner){
    if(isSpinner){
        document.getElementById("spinner-container").classList.remove("d-none");
        document.getElementById("spinner").classList.add("spinner-border");
    }else{
        document.getElementById("spinner-container").classList.add("d-none");
        document.getElementById("spinner").classList.remove("spinner-border");
    }
}