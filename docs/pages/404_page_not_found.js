import { PageNotFound } from "../components/404.js";

export function NotFoundPage() {
    document.getElementById('app').innerHTML = `
        ${PageNotFound()}
    `;

}