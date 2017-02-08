\name{initPHMM}
\alias{initPHMM}

\title{
  Random Initialization for a Hidden Markov Model with Discrete Emissions
}

\description{
Function used to generate a hidden Markov model with random parameters. The code for the methods with categorical values or continuous data can be viewed in \code{"\link{initHMM}"} and \code{"\link{initGHMM}"}, respectively.  
}

\usage{
initPHMM(n)
}

\arguments{
  \item{n}{it sets the number of hidden states to use.}
}

\value{
  A \code{"\link{list}"} that contains all the required values to especify the model.
  \item{Model}{it especifies that the observed values are modeled by a Poisson mixture model.}
  \item{StateNames}{the set of hidden state names.}
  \item{A}{the transition matrix.} 
  \item{B}{a vector that has the lambda parameter of each Poisson distribution.} 
  \item{Pi}{the initial probability vector.}
}

\references{
  Cited references are listed on the \link{RcppHMM} manual page.
}

\examples{
  n <- 2
  model <- initPHMM(n)
  print(model)
}

\keyword{initialization}
