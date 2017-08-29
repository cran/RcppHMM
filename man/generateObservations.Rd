\name{generateObservations}
\alias{generateObservations}

\title{
Generate observations given a model
}

\description{
Function used to generate observations given a hidden Markov model.
}

\usage{
generateObservations(hmm, length)
}

%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{hmm}{a list that contains all the necesary variables to define a hidden Markov model.}
  \item{length}{a value that sets how many observations will be generated.}
}

\value{
A \code{"\link{list}"} that contains the generated observations and the hidden state that generated it.
  \item{X}{a vector with path of hidden states traversed.}
  \item{Y}{a vector with the generated observations.}
}

\examples{
# Set the model parameters
n <- c("First","Second")
m <- c("A","T","C","G")
A <- matrix(c(0.8,0.2,
              0.1,0.9),
            nrow = 2,
            byrow = TRUE)

B <- matrix(c(0.2, 0.2, 0.3, 0.3,
              0.4, 0.4, 0.1, 0.1),
            nrow = 2,
            byrow = TRUE)

Pi <- c(0.5, 0.5)

params <- list( "Model" = "HMM",
                "StateNames" = n,
                "ObservationNames" = m,
                "A" = A,
                "B" = B,
                "Pi" = Pi)

HMM <- verifyModel(params)

# Data simulation
set.seed(100)
length <- 100
observationSequence <- generateObservations(HMM, length)
}

\keyword{ methods }% use one of  RShowDoc("KEYWORDS")
