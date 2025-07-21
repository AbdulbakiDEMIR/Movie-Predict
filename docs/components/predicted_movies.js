import { RenderSelectCategories } from './select_categories.js';
import { spinner } from '../js/script.js';

window.predictedMovies = [];
window.sendCategoriesOpen = function(){
    RenderSelectCategories()
}
async function fetchMoviePosterUrl(movieTitle) {
  try {
    const response = await fetch(`https://abdulbakidemir.com.tr/api_movie_ai/movie_info.php?movie_name=${encodeURIComponent(movieTitle)}`);
    if (!response.ok) {
      const errorJson = await response.json();
      throw new Error(`HTTP ${response.status}: ${JSON.stringify(errorJson)}`);
    }
    const recommendedMovies = await response.json();
    if (recommendedMovies?.poster_url) {
      return recommendedMovies.poster_url;
    } else {
      console.warn("Poster bulunamadı veya geçersiz veri.");
      return null;
    }
  } catch (error) {
    console.error("Film bilgisi alınırken hata oluştu:", error);
    return null;
  }
}
window.predictNext = function(){
    const movieList = document.querySelector(".movie-list");
    const index= parseInt(movieList.getAttribute("value"), 10);
    document.getElementById("btn-next").disabled = false;
    document.getElementById("btn-previous").disabled = false;
    movieList.setAttribute("value",index+1);
    movieList.innerHTML = `
        ${MoviePredictCard(window.predictedMovies[index+1])}
    `
    document.getElementById("movie-location").innerHTML = index+2+"/"+window.MOVIE_SIZE
    if(index+2 === window.predictedMovies.length){
        document.getElementById("btn-next").disabled = true;
    } 
}

window.predictPrevious = function(){
    const movieList = document.querySelector(".movie-list");
    const index= parseInt(movieList.getAttribute("value"), 10);
    document.getElementById("btn-next").disabled = false;
    document.getElementById("btn-previous").disabled = false;
    movieList.setAttribute("value",index-1);
    movieList.innerHTML = `
        ${MoviePredictCard(window.predictedMovies[index-1])}
    `
    document.getElementById("movie-location").innerHTML = index+"/"+window.MOVIE_SIZE

    if(index-1 === 0){
        document.getElementById("btn-previous").disabled = true;
    }
}

window.goToMainPage = function(){
    window.userSelectedCategories = [];
    window.movieGenreMap = {};
    window.userSelectedCategories = []
    window.predictedMovies = []
    changePage('');
}


export function MoviePredictCard(movie){
    return `
        <div class="movie-container mx-auto w-50">
            <div class="movie-image ratio ratio-1x1">
                <img src="${movie.poster}" alt="${movie.title}" class="image">
                <div class="movie-details">
                    <h3>${movie.title}</h3>
                </div>
                <div class="position-relative">
                    <div class="tmdb-icon">
                        <img src="https://www.themoviedb.org/assets/2/v4/logos/v2/blue_long_2-9665a76b1ae401a510ec1e0ca40ddcb3b0cfe45f1d51b77a308fea0845885648.svg">
                    </div>
                </div>
                
            </div>
        </div>
    `;
}

export function MoviePredictComponent(movies){
    return`
        <div id="main-bg">
            <div class="full-center">
                <div class="col-11 col-sm-9 col-md-7 col-lg-6 col-xxl-4 transparent-card">
                    <h1 class="main-title"></h1>
                    <div class="movie-list" value="${movies[0].index}">
                        ${MoviePredictCard(movies[0])}
                    </div>
                    <p id="movie-location"class="text-center">1/${window.MOVIE_SIZE}</p>
                    <div id="movie-previous-next" class="d-flex justify-content-between">
                        <button onclick="predictPrevious()" id="btn-previous" class="btn-sm btn btn-danger" disabled>Previous</button>
                        <button onclick="predictNext()" id="btn-next" class="btn-sm btn btn-danger">Next</button>
                    </div>
                </div>
                <button class="btn btn-primary mt-3" onclick="goToMainPage()">Go to main page</button>
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

export async function  RenderMoviePredictComponent(predict_movies){
    spinner(true);
    var a = 0;
    for (const element of predict_movies) {
        element.movie_id.index = a;
        element.movie_id.poster = await fetchMoviePosterUrl(element.movie_id.title);
        // element.movie_id.poster = "";
        window.predictedMovies.push(element.movie_id)
        a++;
    }
    document.getElementById('app').innerHTML = `   
        ${MoviePredictComponent(window.predictedMovies)}
    `;
    spinner(false);
}