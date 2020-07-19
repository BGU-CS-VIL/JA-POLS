// Copyright 2014, Max Planck Society.
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// http://opensource.org/licenses/BSD-3-Clause)


/*!@file
 * This file contains an application of the Trimmed Grassmann Average to the frames of a movie, in order to compute
 * the meaningful first basis components in a robust manner. You may adapt the code to your needs/data by modifying
 * the function @c number2filename, which from the index of the frame returns a full path of the file of this frame.
 * To limit the memory footprint, the data is stored directly in the temporary memory of the algorithm as it is loaded
 * (see @c iterator_on_image_files). An observer flushes the results to the disk as they arrive from the algorithm (see 
 * @c grassmann_pca_observer).
 */


#include <cstdio>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <include/grassmann_pca_with_trimming.hpp>
#include <include/private/boost_ublas_external_storage.hpp>
#include <include/private/boost_ublas_row_iterator.hpp>

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include <boost/iterator/iterator_facade.hpp>

#include <string>
#include <fstream>

#include "ndarray.h"


#ifndef MAX_PATH
  // "funny" differences win32/posix
  #define MAX_PATH PATH_MAX
#endif

static std::string projectpath = "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/Learn_Piecewise_Subspace/TGA-PCA/";
static std::string movielocation = "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/Learn_Piecewise_Subspace/TGA-PCA/movie";
static std::string eigenvectorslocation = "./";


// on server:
//static std::string projectpath = "/vildata/tohamy/bg_model/Learn_Piecewise_Subspace/TGA-PCA/";
//static std::string movielocation = "/vildata/tohamy/bg_model/Learn_Piecewise_Subspace/TGA-PCA/movie";


namespace grassmann_averages_pca
{
  namespace applications
  {

    void number2filename(size_t file_number, char *filename)
    {
      const size_t dir_num = file_number / 10000;
      sprintf(filename, 
              (projectpath + "movie/starwars_%.3d/frame%d.jpg").c_str(), 
              dir_num, 
              file_number);
    }

        //! An iterator that will load the images on demand instead of storing everything on memory
    template <class T>
    class iterator_on_image_files : 
      public boost::iterator_facade<
            iterator_on_image_files<T>
          , boost::numeric::ublas::vector<T>
          , std::random_access_iterator_tag
          , boost::numeric::ublas::vector<T> const& // const reference
        >
    {
    public:
      typedef iterator_on_image_files<T> this_type;
      typedef boost::numeric::ublas::vector<T> image_vector_type;
      iterator_on_image_files() : m_index(std::numeric_limits<size_t>::max()) 
      {}

      explicit iterator_on_image_files(size_t index)
        : m_index(index) 
      {}

    private:
      friend class boost::iterator_core_access;

      typename this_type::difference_type distance_to(this_type const& r) const
      {
        return typename this_type::difference_type(r.m_index) - typename this_type::difference_type(m_index); // sign promotion
      }

      void increment() 
      { 
        m_index++; 
        image_vector.resize(0, false);
      }

      bool equal(this_type const& other) const
      {
        return this->m_index == other.m_index;
      }

      image_vector_type const& dereference() const 
      { 
        if(image_vector.empty())
        {
          read_image();
        }
        return image_vector; 
      }
  
  
      void read_image() const
      {
        char filename[MAX_PATH];
        number2filename(m_index, filename);
    
        if((m_index % 1) == 0)
        {
          lock_t guard(internal_mutex);
          std::cout << "[THREAD  " << boost::this_thread::get_id() << "] Reading " << filename << std::endl;
        }
    
        cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
        if(!image.data)
        {
          std::ostringstream o;
          o << "error: could not load image '" << filename << "'";
          std::cerr << o.str() << std::endl;
          throw std::runtime_error(o.str());
        }

        const int w = image.size().width;
        const int h = image.size().height;

        image_vector.resize(w * h * 3);
        typename boost::numeric::ublas::vector<T>::iterator it = image_vector.begin();

        for(int y = 0; y < h; y++)
        {
          for(int x = 0; x < w; x++)
          {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            *it++ = pixel[2];
            *it++ = pixel[1];
            *it++ = pixel[0];
          }
        }
      }

      void advance(typename this_type::difference_type n)
      {
        if(n < 0)
        {
          assert((-n) <= static_cast<typename this_type::difference_type>(m_index));
          m_index -= n;
        }
        else
        {
          //assert(n + index <= matrix->size1());
          m_index += n;
        }
        if(n != 0)
        {
          image_vector.resize(0, false);
        }
      }  

      size_t m_index;
      mutable boost::numeric::ublas::vector<T> image_vector;

      // for being able to log in a thread safe manner
      typedef boost::recursive_mutex mutex_t;
      typedef boost::lock_guard<mutex_t> lock_t;

      static mutex_t internal_mutex;


    };

    template <class T>
    typename iterator_on_image_files<T>::mutex_t iterator_on_image_files<T>::internal_mutex;

    template <class data_t>
    struct grassmann_pca_observer
    {
    private:
      size_t element_per_line_during_save;
      size_t last_nb_iteration;
      std::string filename_from_template(std::string template_, size_t i) const
      {
        char filename[MAX_PATH];
        sprintf(filename, template_.c_str(), i);
        return filename;
      }

      void save_vector(const data_t& v, std::string filename) const
      {
        std::ofstream f(filename);
        if(!f.is_open())
        {
          std::cerr << "[ERROR] Cannot open the file " << filename << " for writing" << std::endl;
          return;
        }
    
        std::cout << "-\tWriting file " << filename;
    
        typedef typename data_t::const_iterator element_iterator;
    
        element_iterator itelement(v.begin());
        for(int i = 0; i < v.size(); i++, ++itelement)
        {
          if((i + 1) % element_per_line_during_save == 0)
          {
            f << std::endl;
          }
      
          f << *itelement << " ";
        }
    
        f.close();
        std::cout << " -- done" << std::endl;
      }

    public:
      grassmann_pca_observer(size_t element_per_line_during_save_) : 
        element_per_line_during_save(element_per_line_during_save_)
      {}


      void log_error_message(const char* message) const
      {
        std::cout << message << std::endl;
      }


      //! This is called after centering the data in order to keep track 
      //! of the mean of the dataset
      void signal_mean(const data_t& mean) const
      {
        std::cout << "* Mean computed" << std::endl;
        save_vector(mean, (projectpath + "subspaces/mean_vector.txt").c_str());
      }

      //! Called after the computation of the PCA
      void signal_pca(const data_t& pca,
                      size_t current_eigenvector_dimension) const
      {
        std::cout << "* PCA subspace " << current_eigenvector_dimension << " computed" << std::endl;
        save_vector(pca, filename_from_template((projectpath + "subspaces/vector_pca_%.7d.txt").c_str(), current_eigenvector_dimension));
      }

      //! Called each time a new eigenvector is computed
      void signal_eigenvector(const data_t& current_eigenvector, 
                              size_t current_eigenvector_dimension) const
      {
        std::cout << "* Eigenvector subspace " << current_eigenvector_dimension << " computed in # " << last_nb_iteration << " iterations " << std::endl;
        save_vector(current_eigenvector, filename_from_template((projectpath + "subspaces/vector_subspace_%.7d.txt").c_str(), current_eigenvector_dimension));
      }

      //! Called at every step of the algorithm, at the end of the step
      void signal_intermediate_result(
        const data_t& current_eigenvector_state, 
        size_t current_eigenvector_dimension,
        size_t current_iteration_step) 
      {
        last_nb_iteration = current_iteration_step;
        if((current_iteration_step % 100) == 0)
        {
          std::cout << "* Trimming subspace " << current_eigenvector_dimension << " @ iteration " << current_iteration_step << std::endl;
        }
      }

    };

    int TGA_main_function(float trimming_percentage, size_t num_frames, size_t max_dimension, size_t max_iterations, size_t nb_pca_steps, int nb_processors)
    {
        namespace po = boost::program_options;

        // Parameter values:
        std::cout << "trimming_percentage not set, setting it to " << trimming_percentage << "\n";        
        std::cout << "Nb of frames not set, setting it to " << num_frames << "\n";
        std::cout << "Nb of components not set, setting it to " << max_dimension << "\n";
        std::cout << "Maximum number of iterations # " << max_iterations << "\n";
        std::cout << "Number of PCA steps not set, setting it to " << nb_pca_steps << "\n";
        std::cout << "Number of processors # " << nb_processors << "\n";


        namespace ub = boost::numeric::ublas;
        using namespace grassmann_averages_pca;
        using namespace grassmann_averages_pca::applications;
        using namespace grassmann_averages_pca::details::ublas_helpers;

        // Reading the first image to have the dimensions
        size_t rows(0);
        size_t cols(0);
         
        {
          char filename[MAX_PATH];
          number2filename(0, filename);

          cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
          cv::Size image_size = image.size();
          rows = image_size.height;
          cols = image_size.width;
        }
        
        
        // type of the scalars manipulated by the algorithm
        typedef float input_array_type;


        // Allocate data
        std::cout << "=== Allocate ===" << std::endl;
        std::cout << "  Number of images: " << num_frames << std::endl;
        std::cout << "  Image size:       " << cols << "x" << rows << " (RGB)" << std::endl;
        std::cout << "  Data size:        " << (num_frames*rows*cols * 3 * sizeof(input_array_type)) / (1024 * 1024) << " MB" << std::endl;
          
        iterator_on_image_files<float> iterator_file_begin(0);
        iterator_on_image_files<float> iterator_file_end(num_frames + 0);


        // type of the data extracted from the input iterators
        typedef ub::vector<input_array_type> data_t;
        // type of the observer
        typedef grassmann_pca_observer<data_t> observer_t;
        // type of the trimmed grassmann algorithm
        typedef grassmann_pca_with_trimming< data_t, observer_t > grassmann_pca_with_trimming_t;


        // main instance
        grassmann_pca_with_trimming_t instance(trimming_percentage / 100);
        
        
        typedef std::vector<data_t> output_eigenvector_collection_t;
        output_eigenvector_collection_t v_output_eigenvectors(max_dimension);


        if(nb_processors > 0)
        {
          if(!instance.set_nb_processors(nb_processors))
          {
            std::cerr << "[configuration]" << "Incorrect number of processors. Please consult the documentation (was " << nb_processors << ")" << std::endl;
            return 1;
          }
        }



        if(!instance.set_nb_steps_pca(nb_pca_steps))
        {
          std::cerr << "[configuration]" << "Incorrect number of regular PCA steps. Please consult the documentation (was " << nb_pca_steps << ")" << std::endl;
          return 1;
        }

        // setting the observer
        observer_t my_simple_observer(cols);
        if(!instance.set_observer(&my_simple_observer))
        {
          std::cerr << "[configuration]" << "Error while setting the observer" << std::endl;
          return 1;
        }

        // requesting the centering of the data
        if(!instance.set_centering(true))
        {
          std::cerr << "[configuration]" << "Error while configuring the centering" << std::endl;
          return 1;
        }

        // running the computation
        bool ret = instance.batch_process(
          max_iterations,
          max_dimension,
          iterator_file_begin,
          iterator_file_end,
          v_output_eigenvectors.begin(),
          projectpath);

        // Results are saved in the observer instance

        if(!ret)
        {
          std::cerr << "The process returned an error" << std::endl;
          return 1;
        }

        return 0;

    }
  }
}


extern "C"{
  int TGA_main_function(float trimming_percentage, size_t num_frames, size_t max_dimension, size_t max_iterations, size_t nb_pca_steps, int nb_processors);
}



// ---------------------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------------------------------------


int main(int argc, char *argv[])
{
  float trimming_percentage = 50;
  size_t num_frames = atoi(argv[1]);
  size_t max_dimension = 5;
  size_t max_iterations = num_frames;
  size_t nb_pca_steps = 3;
  int nb_processors = 0;

  grassmann_averages_pca::applications::TGA_main_function(trimming_percentage, num_frames, max_dimension, max_iterations, nb_pca_steps, nb_processors);
}

