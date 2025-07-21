import { getRecommendedMoviesByCategories } from '../js/component_movie_card.js';
import { selectRandomUniqueMoviesWithPosters } from '../js/component_movie_card.js';
import { RenderSelectCategories } from './select_categories.js';
import { mlb_classes } from '../model/mlb_classes.js';
import { movies } from '../model/movies.js';
import { spinner } from '../js/script.js';

window.sendCategoriesOpen = function(){
    RenderSelectCategories()
}
window.next = function(){
    const movieList = document.querySelector(".movie-list");
    const index= parseInt(movieList.getAttribute("value"), 10);
    document.getElementById("btn-next").disabled = false;
    document.getElementById("btn-previous").disabled = false;
    movieList.setAttribute("value",index+1);
    movieList.innerHTML = `
        ${MovieCard(window.recommendedMovies[index+1])}
    `
    document.getElementById("movie-location").innerHTML = index+2+"/"+window.MOVIE_SIZE
    rating(window.recommendedMovies[index+1])
    if(index+2 === window.recommendedMovies.length){
        document.getElementById("btn-next").disabled = true;
        document.getElementById("show-rating-btn").disabled = false;
    } 
}

window.previous = function(){
    const movieList = document.querySelector(".movie-list");
    const index= parseInt(movieList.getAttribute("value"), 10);
    document.getElementById("btn-next").disabled = false;
    document.getElementById("btn-previous").disabled = false;
    movieList.setAttribute("value",index-1);
    movieList.innerHTML = `
        ${MovieCard(window.recommendedMovies[index-1])}
    `
    document.getElementById("movie-location").innerHTML = index+"/"+window.MOVIE_SIZE

    rating(window.recommendedMovies[index-1])
    if(index-1 === 0){
        document.getElementById("btn-previous").disabled = true;
    }
}
export function MovieCard(movies){
    return `
        <div class="movie-container mx-auto w-50">
            <div class="movie-image ratio ratio-1x1">
                <img src="${movies.poster}" alt="${movies.title}" class="image">
                <div class="movie-details">
                    <h3>${movies.title}</h3>
                </div>
                <div class="position-relative">
                    <div class="tmdb-icon">
                        <img src="https://www.themoviedb.org/assets/2/v4/logos/v2/blue_long_2-9665a76b1ae401a510ec1e0ca40ddcb3b0cfe45f1d51b77a308fea0845885648.svg">
                    </div>
                </div>
            </div>
            <div id="rating${movies.movie_id}"  value="${movies.rating}" class="rating-stars">
                <i class="fa-solid fa-star"></i>
                <i class="fa-regular fa-star"></i>
                <i class="fa-regular fa-star"></i>
                <i class="fa-regular fa-star"></i>
                <i class="fa-regular fa-star"></i>
            </div>
        </div>
    `;
}

export function MovieRatingComponent(){
    return`
        <div id="main-bg">
            <div class="full-center">
                <div class="col-11 col-sm-9 col-md-7 col-lg-6 col-xxl-4 transparent-card">
                    <h1 class="main-title">Rating</h1>
                    <div class="movie-list" value="${window.recommendedMovies[0].index}">
                        ${MovieCard(window.recommendedMovies[0])}
                    </div>
                    <p id="movie-location"class="text-center">1/${window.MOVIE_SIZE}</p>
                    <div id="movie-previous-next" class="d-flex justify-content-between">
                        <button onclick="previous()" id="btn-previous" class="btn-sm btn btn-danger" disabled>Previous</button>
                        <button onclick="next()" id="btn-next" class="btn-sm btn btn-danger">Next</button>
                    </div>
                    <div class="w-100 d-flex justify-content-between mt-3">
                        <button onclick="sendCategoriesOpen()" class="btn btn-warning">Select Category</button>
                        <button id="show-rating-btn"class="btn btn-primary" onclick="showRating()" disabled>Show Rating</button>
                    </div>
                </div>
            </div>
        </div>
        <footer style="background-color: #111; color: #bbb; padding: 20px 0; text-align: center; font-size: 14px;">
            <div style="max-width: 800px; margin: auto;">
                <p>This website is developed using the <strong>The Movie Database (TMDB)</strong> API.</p>
                <a href="https://www.themoviedb.org/" target="_blank" style="display: inline-block; margin: 10px 0;">
                <img src="https://www.themoviedb.org/assets/2/v4/logos/v2/blue_square_2-d537fb228cf3ded904ef09b136fe3fec72548ebc1fea3fbbd1ad9e36364db38b.svg" 
                    alt="TMDB Logo" width="80" style="opacity: 0.8; transition: opacity 0.3s;">
                </a>
            </div>
        </footer>
    `
}

export async function  RenderMovieRatingComponent(){
    spinner(true);
    window.filteredMovies = await getRecommendedMoviesByCategories(movies, window.userSelectedCategories);
    window.recommendedMovies = await selectRandomUniqueMoviesWithPosters(window.filteredMovies, MOVIE_SIZE);
    window.genreClasses = mlb_classes;
    window.movieGenreMap = {};
    window.recommendedMovies.forEach(item => {
        const movie = window.filteredMovies.find(m => m.movieId === item.movie_id);
        if (movie) {
            window.movieGenreMap[item.movie_id] = movie.genres;
        }
    });
    document.getElementById('app').innerHTML = `   
        ${MovieRatingComponent()}
    `;
    rating(window.recommendedMovies[0]);
    spinner(false);
}