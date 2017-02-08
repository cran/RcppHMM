//=================================
// include guard
#pragma once

//=================================
// forward declared dependencies

//=================================
// included dependencies
#include "vHMM.h"

//=================================
// the actual class
class HMMcont : public vHMM  // Parent object
{
    //--------------------------------------------------------------------------
    // CONSTRUCTOR & DESTRUCTOR:
    //--------------------------------------------------------------------------

public:

    //! Constructor of HMMcont.
    HMMcont(unsigned short int  numberStates);
    HMMcont(Rcpp::CharacterVector stateNames);
    HMMcont(Rcpp::CharacterVector stateNames, Rcpp::NumericMatrix A, Rcpp::NumericMatrix B, Rcpp::NumericVector Pi);    

    //! Destructor of HMMcont.
    virtual ~HMMcont(void);

    //--------------------------------------------------------------------------
    // PUBLIC METHODS:
    //--------------------------------------------------------------------------

public:    
    //Getters
    Rcpp::NumericMatrix getB(void) const;
    
    // Setters
    // hidden states -> rows    
    void setB(Rcpp::NumericMatrix B);
    void setParameters(Rcpp::NumericMatrix A, Rcpp::NumericMatrix B, Rcpp::NumericVector  Pi);

    // Evaluation methods 
    double evaluation(Rcpp::NumericVector sequence, char method = 'f');

    // Decoding methods
    Rcpp::CharacterVector viterbi(Rcpp::NumericVector sequence);
    Rcpp::CharacterVector forwardBackward(Rcpp::NumericVector sequence);

    // Learning methods 
    double loglikelihood(Rcpp::NumericMatrix sequence);
    void learnBW(Rcpp::NumericVector sequences, unsigned short int iter = 100, double delta = EPSILON, unsigned char pseudo = 0, bool print = true);
    void learnEM(Rcpp::NumericMatrix sequences, unsigned short int iter = 100, double delta = EPSILON, unsigned char pseudo = 0, bool print = true);
    //void learnEMParallel(Rcpp::NumericMatrix sequences, unsigned short int iter = 100, double delta = EPSILON, unsigned char pseudo = 0, bool print = true);

    // Simulation
    Rcpp::List generateObservations(unsigned short int length);

    // Miscellaneous
    Rcpp::List toList(void) const;
    //std::ostream& print(std::ostream& out) const; 

    //--------------------------------------------------------------------------
    // PROTECTED METHODS:
    //--------------------------------------------------------------------------

protected:
    //  Miscellaneous
    void randomInit(double min, double max);   

    //--------------------------------------------------------------------------
    // PRIVATE METHODS:
    //--------------------------------------------------------------------------

private:
    // Evaluation methods
    void  forwardMatrix(Rcpp::NumericVector sequence, unsigned int length , scaledMatrix & forward);
    void  backwardMatrix(Rcpp::NumericVector sequence, unsigned int length , scaledMatrix & backward);

    // Decoding methods
    Rcpp::NumericMatrix forwardBackwardGamma(Rcpp::NumericVector sequence);
    void forwardBackwardGamma(Rcpp::NumericVector index, scaledMatrix & forward, scaledMatrix & backward,  Rcpp::NumericVector & scaledf, Rcpp::NumericVector & scaledb, Rcpp::NumericMatrix & matrix, unsigned int length);

    // Learning methods  
    void BaumWelch( Rcpp::NumericVector sequences, unsigned int pseudo);  
    void expectationMaximization(Rcpp::NumericMatrix sequences, unsigned int pseudo);
    //void expectationMaximizationParallel( Rcpp::NumericMatrix sequences, unsigned int pseudo);

    //--------------------------------------------------------------------------
    // PRIVATE MEMBERS:
    //--------------------------------------------------------------------------

private:
    Rcpp::NumericMatrix m_B;  // Emission matrix         

};
