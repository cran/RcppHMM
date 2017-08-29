\name{learnEM}
\alias{learnEM}

\title{
Expectation-Maximization algorithm to estimate the model parameters
}

\description{
Expectation-Maximization algorithm to estimate the model parameters based on a single or multiple observed sequences.
}

\usage{
learnEM(hmm, sequences, iter = 100, delta = 1e-05, pseudo = 0, print = TRUE )
}


\arguments{
  \item{hmm}{a list that contains all the necesary variables to define a hidden Markov model.}
  \item{sequences}{a matrix that contains in each row a sequence of observations to be evaluated.}
  \item{iter}{a value that sets the maximum number of iterations to be done.}
  \item{delta}{a value set to be the minimum error considered as a convergence criteria.}
  \item{pseudo}{a value set to consider pseudo-counts.}
  \item{print}{a logical value to print the errar at each iteration.}
}

\value{
A \code{"\link{list}"} that contains the estimated hidden Markov model parameters.
}

\references{
Cited references are listed on the \link{RcppHMM} manual page.
}


\seealso{
\code{\link{generateObservations}} , \code{\link{verifyModel}}
}

\examples{
# Set the model parameters to be estimated
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
seqs <- 100
observationSequences<- c()
for(i in 1:seqs){
  Y <- generateObservations(HMM , length)$Y
  observationSequences <- rbind(observationSequences , Y)
}

dim(observationSequences)
table(observationSequences)

# New model random initialization

set.seed(1000)
newModel <- initHMM(2,4) 
n = c("X1","X2")
m = c("A","T","C","G")
newModel <- setNames(newModel,
                        list( "StateNames" = n,
                              "ObservationNames" = m) )

\donttest{
newModel <- learnEM(newModel,
        observationSequences,
        iter=300, 
        delta = 1E-8,
        pseudo = 3,
        print = TRUE)
}

print(newModel)   

}

\keyword{optimize}
