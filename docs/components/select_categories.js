import { RenderMovieRatingComponent } from "./movie_card.js";
import { mlb_classes } from '../model/mlb_classes.js';
import { spinner } from "../js/script.js";

window.categoryItemClick = function(category){
    const btn = document.getElementById("category-btn");
    const element = document.getElementById("item-"+category);
    const val = element.getAttribute("selected");
    
    if(val === "false"){
        if(window.userSelectedCategories.length === 3){
            
        }else{
            element.setAttribute("selected","true");
            window.userSelectedCategories.push(category)
            btn.disabled = false
        }
        
    }else{
        element.setAttribute("selected","false");
        window.userSelectedCategories = window.userSelectedCategories.filter(e => e !== category);
        if(window.userSelectedCategories.length === 0){
            btn.disabled = true
        }
    }
}

window.sendCategoriesClose = function(){
    spinner(true);
    RenderMovieRatingComponent()
}


function CategoriesItem(category){
    const isSelected = window.userSelectedCategories.includes(category);
    return `
        <div class="category-item-container col px-1 py-2">
            <div onclick="categoryItemClick('${category}')" id="item-${category}" category_value="${category}" selected="${isSelected}" class="category-item btn btn-danger w-100">
                <p class="my-2">${category}</p>
            </div>
        </div>
    `
}

export function SelectCategories(){
    let categoriesContent = '';
    mlb_classes.forEach(element => {
        categoriesContent = categoriesContent + CategoriesItem(element)
    });
    var btnDisable = "disabled";
    if(window.userSelectedCategories.length > 0){
        btnDisable = "";
    }
    return`
        <div id="categories-container" show="true">
            <div class="col-md-5 col-lg-4 mx-auto pt-4 pb-3 px-3 transparent">
                <h3 class="px-3 text-center fw-bold">Select Category</h3>
                <span class="w-100 d-flex justify-content-center">Select up to 3 categories</span>
                <div class="d-flex row row-cols-3 px-2 py-3" full="false">
                    ${categoriesContent}
                </div>
                <div class="w-100 d-flex justify-content-end">
                    <button onclick="sendCategoriesClose()" id="category-btn" class="btn btn-primary" ${btnDisable}>Next</button>
                </div>
            </div>
        </div>
    `

    
}

export function RenderSelectCategories(){
    spinner(true);
    document.getElementById('app').innerHTML = `   
        ${SelectCategories()}
    `;
    spinner(false);
}