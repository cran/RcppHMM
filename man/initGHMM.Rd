\name{initGHMM}
\alias{initGHMM}

\title{
Random Initialization for a Hidden Markov Model with Continuous Emissions
}

\description{
Function used to generate a hidden Markov model with random parameters. The code for the methods with categorical values or discrete data can be viewed in \code{"\link{initHMM}"} and \code{"\link{initPHMM}"}, respectively.  
}

\usage{
initGHMM(n)
}

\arguments{
  \item{n}{it sets the number of hidden states to use.}
}

\value{
A \code{"\link{list}"} that contains all the required values to especify the model.
\item{Model}{it especifies that the observed values are modeled by a Gaussian mixture model.}
\item{StateNames}{the set of hidden state names.}
\item{A}{the transition matrix.} 
\item{B}{a matrix that has the parameters of each Normal distribution, in the first column the mean and in the second the standard deviation.} 
\item{Pi}{the initial probability vector.}
}

\references{
Cited references are listed on the \link{RcppHMM} manual page.
}

\examples{
n <- 2
model <- initGHMM(n)
print(model)
}

\keyword{initialization}
