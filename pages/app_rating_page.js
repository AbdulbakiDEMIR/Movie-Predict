import { SelectCategories } from '../components/select_categories.js';
import { spinner } from '../js/script.js';
window.MOVIE_SIZE = 10;

window.model = await tf.loadGraphModel('model/final_model/model.json');

window.userSelectedCategories = [];
window.movieGenreMap = {};

export function AppRatingPage() {
    spinner(true);
    document.getElementById('app').innerHTML = `   
        ${SelectCategories()}
    `;
    spinner(false);
}
