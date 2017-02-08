\name{setNames}
\alias{setNames}

\title{
Set the names of the model
}

\description{
Function used to set new hidden state names to the model. If it is a categorical model, it also set the observation names. This function verifies that all the parameters size agree with the provided new names.  
}

\usage{
  setNames(hmm , names)
}

\arguments{
  \item{hmm}{a list that contains all the necesary variables to define a hidden Markov model.}
  \item{names}{a list with the new names to set in the model.}
}

\value{
  A \code{"\link{list}"} that contains the verified hidden Markov model parameters.
}

\examples{
## Values for a hidden Markov model with categorical observations

set.seed(1000)
newModel <- initHMM(2,4)
n <- c("First","Second")
m <- c("A","T","C","G")
newModel <- setNames(newModel,
                    list( "StateNames" = n,
                          "ObservationNames" = m) )

## Values for a hidden Markov model with continuous observations

set.seed(1000)
newModel <- initGHMM(3) 
n <- c("Low",  "Normal", "High" )
newModel <- setNames(newModel,
                     list( "StateNames" = n))

## Values for a hidden Markov model with discrete observations

set.seed(1000)
newModel <- initPHMM(3) 
n <- c("Low",  "Normal", "High" )
newModel <- setNames(newModel,
                     list( "StateNames" = n))

}

\keyword{methods}