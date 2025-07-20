// config.js
export const config =  {
    // basePath: "/model-javascript/",
    basePath: "/",
    scripts: [
        { path: "js/script.js", type: "module" },
        { path: "app.js", type: "module" }
    ],
    styles: [
        { path: "css/style.css" }
    ]
};

export const pageQueryParameterKey = "page";