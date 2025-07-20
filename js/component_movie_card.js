import { movie_encoder } from '../model/movie_encoder.js';
import { MovieCard } from '../components/movie_card.js';
import { RenderMoviePredictComponent } from '../components/predicted_movies.js';
import { spinner } from './script.js';

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

export function encodeMovieId(rawId) {
  return movie_encoder.indexOf(rawId);
}

function filterMoviesBySelectedCategories(moviesList, selectedCategories) {
  if (selectedCategories.length === 0) return moviesList;
  return moviesList.filter(movie => {
    return selectedCategories.some(category => movie.genres.includes(category));
  });
}
export async function getRecommendedMoviesByCategories(allMovies, categories) {
  const filtered = filterMoviesBySelectedCategories(allMovies, categories);
  return filtered;
}

export async function selectRandomUniqueMoviesWithPosters(filteredMovies, count) {
  const selectedMovieIds = new Set();
  const moviesWithDetails = [];

  while (selectedMovieIds.size < count && selectedMovieIds.size < filteredMovies.length) {
    const randomIndex = Math.floor(Math.random() * filteredMovies.length);
    const movie = filteredMovies[randomIndex];
    const movieId = movie.movieId;

    if (!selectedMovieIds.has(movieId)) {
      selectedMovieIds.add(movieId);

      // const posterUrl = "";
      const posterUrl = await fetchMoviePosterUrl(movie.title);

      moviesWithDetails.push({
        index: moviesWithDetails.length,
        movie_id: movieId,
        rating: 0,
        title: movie.title,
        year: movie.year,
        genres: movie.genres,
        poster: posterUrl
      });
    }
  }

  return moviesWithDetails;
}

function makeGenreVector(genres) {
  if (!Array.isArray(genres)) {
    console.warn("⚠️ genres değeri geçersiz:", genres);
    return new Array(window.genreClasses.length).fill(0);
  }
  return window.genreClasses.map(g => genres.includes(g) ? 1 : 0);
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

window.rating = function(movierecommendedMovies){
    document.querySelectorAll( '#rating'+movierecommendedMovies.movie_id+' i').forEach((star, index) => {
        star.addEventListener('click', () => {
            const item = window.recommendedMovies.find(obj => obj.index === movierecommendedMovies.index);
            if (item) {
            item.rating = index;
            }
            document.getElementById('rating'+movierecommendedMovies.movie_id).setAttribute('value', index);
            const stars = document.querySelectorAll('#rating'+movierecommendedMovies.movie_id+' i');
            stars.forEach((s, i) => {
                s.classList.toggle('fa-solid', i <= index);
                s.classList.toggle('fa-regular', i > index);
            });
        });
    });
    document.querySelectorAll( '#rating'+movierecommendedMovies.movie_id+' i').forEach((star, index) => {
        star.addEventListener('mouseover', () => {
            const stars = document.querySelectorAll('#rating'+movierecommendedMovies.movie_id+' i');
            stars.forEach((s, i) => {
                s.classList.toggle('fa-solid', i <= index);
                s.classList.toggle('fa-regular', i > index);
            });
        });
    });
    document.querySelectorAll( '#rating'+movierecommendedMovies.movie_id+' i').forEach((star, index) => {
        star.addEventListener('mouseout', () => {
            index = parseInt(document.getElementById('rating'+movierecommendedMovies.movie_id).getAttribute('value'));
            const stars = document.querySelectorAll('#rating'+movierecommendedMovies.movie_id+' i');
            stars.forEach((s, i) => {
                s.classList.toggle('fa-solid', i <= index);
                s.classList.toggle('fa-regular', i > index);
            });
        });
    });
    const index = parseInt(document.getElementById('rating'+movierecommendedMovies.movie_id).getAttribute('value'));
    const stars = document.querySelectorAll('#rating'+movierecommendedMovies.movie_id+' i');
    stars.forEach((s, i) => {
        s.classList.toggle('fa-solid', i <= index);
        s.classList.toggle('fa-regular', i > index);
    });
}

window.showRating = async function () {
    spinner(true);
    const candidateMovies = window.filteredMovies
    .filter(filter_movie => !window.recommendedMovies.some(r => r.movie_id === filter_movie.movieId)) // kullanıcı zaten oylamamışsa
    const predictions = [];

    for (const movieId of candidateMovies) {
        const encodedMovieId = encodeMovieId(movieId);
        const genres = window.movieGenreMap[movieId] || [];
        const genreVector = makeGenreVector(genres);

        const pred = window.model.predict({
        user_input: tf.tensor([[0]]),
        movie_input: tf.tensor([[encodedMovieId]]),
        genre_input: tf.tensor([genreVector])
        });

        const result = await pred.array();
        predictions.push({ movie_id: movieId, score: result[0][0] });
    }
    predictions.sort((a, b) => b.score - a.score);
    const topRecommendations = predictions.slice(0, 10);
    RenderMoviePredictComponent(topRecommendations);
};






