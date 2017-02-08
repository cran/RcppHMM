\name{forwardBackward}
\alias{forwardBackward}

\title{
Forward-backward algortihm for hidden state decoding
}

\description{
Function used to get the most likely hidden states at each observation in the provided sequence.
}

\usage{
forwardBackward(hmm, sequence)
}

\arguments{
  \item{hmm}{a list that contains all the necesary variables to define a hidden Markov model.}
  \item{sequence}{a vector that contains a sequence of observations to be decoded.}
}

\value{
A vector that contains the hidden states traveled that generated the observed sequence.
}

\references{
Cited references are listed on the \link{RcppHMM} manual page.
}

\seealso{
\code{\link{generateObservations}} , \code{\link{verifyModel}} , \code{\link{viterbi}}
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

#Sequence decoding
hiddenStates <- forwardBackward(HMM, observationSequence$Y)
}

\keyword{methods}
