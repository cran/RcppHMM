\name{verifyModel}
\alias{verifyModel}

\title{
Model parameter verification
}

\description{
Function used to verify that all the parameters satisfy the model constraints.
}

\usage{
verifyModel(model)
}

\arguments{
  \item{model}{ a list with the necessary parameters to set a hidden Markov model with fixed values.}
}

\details{
The model must have an stochastic transition matrix and an stochastic initial probability vector, also the row and column sizes must coincide with the number of provided state names. If the model uses categorical values, the emission matrix also must be stochastic and must have a column for each observation name and a row for each state name. And if the model uses discrete data, all the values must be positive.  
}

\value{
  A \code{"\link{list}"} that contains the verified hidden Markov model parameters.
}

\examples{
## Values for a hidden Markov model with categorical observations

n <- c("First","Second")
m <- c("A","T","C","G")
A <- matrix(c(0.378286,0.621714,
              0.830970,0.169030),
            nrow = 2,
            byrow = TRUE)

B <- matrix(c(0.1930795, 0.2753869, 0.3463100, 0.1852237,
              0.2871577, 0.1848870, 0.1614925, 0.3664628),
            nrow = 2,
            byrow = TRUE)

Pi <- c(0.4757797, 0.5242203)


params <- list( "Model" = "HMM",
                "StateNames" = n,
                "ObservationNames" = m,
                "A" = A,
                "B" = B,
                "Pi" = Pi)

verifiedModel <- verifyModel(params)
print(verifiedModel)

## Values for a hidden Markov model with continuous observations

n <- c("Low","Normal","High")

A <- matrix(c(0.5, 0.3,0.2,
              0.2, 0.6, 0.2,
              0.1, 0.3, 0.6),
            ncol=length(n), byrow=TRUE)

B <- matrix(c(0,100,            # First Gaussian with mean 0 and standard deviation 100
              500,300,          # Second Gaussian with mean 500 and standard deviation 300
              1000,200),        # Third Gaussian with mean 1000 and standard deviation 200
            nrow=length(n), byrow=TRUE)

Pi <- rep(1/length(n), length(n))

HMM.real <- verifyModel(list("Model"="GHMM", "StateNames" = n, "A" = A, "B" = B, "Pi" = Pi))
print(HMM.real)

## Values for a hidden Markov model with discrete observations

n <- c("Low","Normal","High")

A <- matrix(c(0.5, 0.3,0.2,
              0.2, 0.6, 0.2,
              0.1, 0.3, 0.6),
            ncol=length(n), byrow=TRUE)

B <- c(2600,  # First distribution with mean 2600
       2700,  # Second distribution with mean 2700
       2800)  # Third distribution with mean 2800

Pi <- rep(1/length(n), length(n))

HMM.discrete <- verifyModel(list("Model"="PHMM", "StateNames" = n, "A" = A, "B" = B, "Pi" = Pi))
print(HMM.discrete)

}

\keyword{methods}
