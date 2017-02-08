\name{setParameters}
\alias{setParameters}

\title{
Set the model parameters
}

\description{
Function used to set the model parameters. This function verifies that all the parameters size agree with the names size.    
}

\usage{
setParameters(hmm , params)
}

\arguments{
  \item{hmm}{a list that contains all the necesary variables to define a hidden Markov model.}
  \item{params}{a list with the new parameters to set in the model.}
}

\value{
  A \code{"\link{list}"} that contains the verified hidden Markov model parameters.
}

\examples{
## Values for a hidden Markov model with categorical observations

set.seed(1000)
newModel <- initHMM(2,4)

A <- matrix(c(0.378286,0.621714,
              0.830970,0.169030),
            nrow = 2,
            byrow = TRUE)

B <- matrix(c(0.1930795, 0.2753869, 0.3463100, 0.1852237,
              0.2871577, 0.1848870, 0.1614925, 0.3664628),
            nrow = 2,
            byrow = TRUE)

Pi <- c(0.4757797, 0.5242203)

newModel <- setParameters(newModel,
                    list( "A" = A,
                          "B" = B,
                          "Pi" = Pi) )

## Values for a hidden Markov model with continuous observations

set.seed(1000)
n <- 3
newModel <- initGHMM(n) 

A <- matrix(c(0.5, 0.3, 0.2,
              0.2, 0.6, 0.2,
              0.1, 0.3, 0.6),
            ncol= n, byrow=TRUE)

B <- matrix(c(0,100,            # First Gaussian with mean 0 and standard deviation 100
              500,300,          # Second Gaussian with mean 500 and standard deviation 300
              1000,200),        # Third Gaussian with mean 1000 and standard deviation 200
            nrow=n, byrow=TRUE)

Pi <- rep(1/n, n)

newModel <- setParameters(newModel,
                    list( "A" = A,
                          "B" = B,
                          "Pi" = Pi) )

## Values for a hidden Markov model with discrete observations

set.seed(1000)
n <- 3
newModel <- initPHMM(n) 

A <- matrix(c(0.5, 0.3,0.2,
              0.2, 0.6, 0.2,
              0.1, 0.3, 0.6),
            ncol=n, byrow=TRUE)

B <- c(2600,  # First distribution with mean 2600
       2700,  # Second distribution with mean 2700
       2800)  # Third distribution with mean 2800

Pi <- rep(1/n , n)


newModel <- setParameters(newModel,
                    list( "A" = A,
                          "B" = B,
                          "Pi" = Pi) )
}

\keyword{methods}
