import { renderApp } from "./routing/routing.js";

export function App() {
    renderApp();
    window.addEventListener("popstate", renderApp);
}

App()
