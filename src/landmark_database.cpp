#include "landmark_database.hpp"

LandmarkDatabase::LandmarkDatabase()
{
}

LandmarkDatabase::LandmarkDatabase(int nb_frames)
{
    m_graph = Graph<std::vector<double>>(nb_frames);
}

void LandmarkDatabase::save(const std::string& filename)
{
    int nb_frames = m_graph.getNbVertices();
    std::vector<double> to_save;
    for (int i=0; i<nb_frames; ++i)
    {
        for (int j=(i+1); j<nb_frames; ++j)
        {
            if (m_graph.is_there_edge(i,j))
            {
                std::vector<double> *list_landmarks_pair = m_graph.get_edge_ptr(i,j);
                int nb_correspondences_pair = ((int)list_landmarks_pair->size())/4;
                for (int k=0; k<nb_correspondences_pair; ++k)
                {
                    to_save.push_back(i);
                    to_save.push_back(j);
                    for (int d=0; d<4; ++d)
                        to_save.push_back((*list_landmarks_pair)[4*k+d]);
                }

            }

        }
    }

    saveVector<double>(to_save,filename);

}

void LandmarkDatabase::load(const std::string& filename)
{
    int nb_frames = this->getNbFrames();
    std::vector<double> list_correspondences;
    loadVector<double>(list_correspondences,filename);
    int nb_correspondences = (int)list_correspondences.size()/6;
    for (int k=0; k<nb_correspondences; ++k)
    {
        int i = list_correspondences[6*k];
        int j = list_correspondences[6*k+1];
        if ((i<nb_frames) && (j<nb_frames))
        {
            if (m_graph.is_there_edge(i,j)==false)
                m_graph.add_edge(i,j,std::vector<double>(0));
            std::vector<double> *edge_ptr =  m_graph.get_edge_ptr(i,j);
            for (int d=2; d<6; ++d)
                edge_ptr->push_back(list_correspondences[6*k+d]);
        }
        else
            std::cout << "Warning! While loading correspondences, the pair (" << i << "," << j << ") exceeds the given number of frames in the database" << std::endl;
    }
}

void LandmarkDatabase::addToDatabase(int input_frame, int output_frame, const std::vector<cv::Point2f>& landmarks_input, const std::vector<cv::Point2f>& landmarks_output)
{
    int nb_landmarks = (int)landmarks_input.size();
    if (nb_landmarks!=(int)landmarks_output.size())
        std::cout << "Error! In addToDatabase: try to add landmarks where the number of input and output landmark is different" <<std::endl;
    else
    {
        if (m_graph.is_there_edge(input_frame,output_frame))
        {
            std::cout << "Warning: Try to add landmarks to an existing pair in the database. We erase the previous entries" << std::endl;
            m_graph.get_edge_ptr(input_frame,output_frame)->clear();
        }
        else
           m_graph.add_edge(input_frame,output_frame,std::vector<double>(0));
        std::vector<double> *edge_ptr = m_graph.get_edge_ptr(input_frame,output_frame);
        for (int p=0; p<nb_landmarks; ++p)
        {
            edge_ptr->push_back(landmarks_input[p].x);
            edge_ptr->push_back(landmarks_input[p].y);
            edge_ptr->push_back(landmarks_output[p].x);
            edge_ptr->push_back(landmarks_output[p].y);
        }
    }
}

void LandmarkDatabase::eraseFromDatabase(int input_frame, int output_frame)
{
    m_graph.remove_edge(input_frame,output_frame);
}

bool LandmarkDatabase::isPairInDatabase(int input_frame, int output_frame) const
{
    return m_graph.is_there_edge(input_frame,output_frame);
}

void LandmarkDatabase::loadLandmarks(int input_frame, int output_frame, std::vector<cv::Point2f>& landmarks_input, std::vector<cv::Point2f>& landmarks_output)
{
    landmarks_input.clear();
    landmarks_output.clear();
    if (m_graph.is_there_edge(input_frame,output_frame)==false)
        std::cout << "Error! Try to load landmarks from a non-existent pair in database" << std::endl;
    else
    {
         std::vector<double> *edge_ptr = m_graph.get_edge_ptr(input_frame,output_frame);
         int nb_correspondences =  ((int)edge_ptr->size())/4;
         landmarks_input.resize(nb_correspondences);
         landmarks_output.resize(nb_correspondences);
         for (int k=0; k<nb_correspondences; ++k)
         {
            landmarks_input[k] = cv::Point2f((*edge_ptr)[4*k],(*edge_ptr)[4*k+1]);
            landmarks_output[k] = cv::Point2f((*edge_ptr)[4*k+2],(*edge_ptr)[4*k+3]);
         }
    }

}





