/*  High performance statistical computing for massive data

- this was written to compare loops and vector operations in R and in C
 to determine where it is appropriate to change to C code in an R script

*/

#include <sys/time.h>
#include <Rcpp.h>

#ifdef USE_OMP
    #include <omp.h>
#endif

double getTimeDiff(struct timeval t1,
                   struct timeval t2) {
    return ((t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec-t1.tv_usec)/1000000.0);
}

namespace Logging {
    
    static long nCtors = 0;
    static long nCopies = 0;
    static long nDtors = 0;
    
    class NumericVector : public Rcpp::NumericVector {
    public:
        NumericVector(size_t n) : Rcpp::NumericVector(n) { 
            nCtors++; 
        }
                
        NumericVector(const Rcpp::NumericVector& v) : Rcpp::NumericVector(v) {
            nCtors++; 
        }
        
        template <typename VectorExpression>
        NumericVector& operator=(const VectorExpression& v) {         
            nCopies++;
            const int n = size();
            for (int i = 0; i < n; ++i) {
                (*this)[i] = v[i];
            }
            return *this;
        }
        
        virtual ~NumericVector() {
            nDtors++;
        }
                
    private:
        NumericVector(); // Disallow default constructor       
    };
    
    static void clearCounters() {
        nCtors = 0; nCopies = 0; nDtors = 0;
    }
    
    static long getCtors() { return nCtors; }
    static long getCopies() { return nCopies; }
    static long getDtors() { return nDtors; }

};

namespace Rcpp {
    // For compatibility with Logging
    static void clearCounters() { }
    static long getCtors() { return 0; }
    static long getCopies() { return 0; }
    static long getDtors() { return 0; }
};

namespace ComponentWiseOps {
    
    // Syntactic sugar to write: A + B
    template <typename NumericVector>
    NumericVector operator+(const NumericVector& lhs, const NumericVector& rhs) {
        const int n = lhs.size();
        NumericVector result(n);        
        for (int i = 0; i < n; ++i) {
            result[i] = lhs[i] + rhs[i];
        }
        return result;
    }
    
    // Syntactic sugar to write: A * B
    template <typename NumericVector>
    NumericVector operator*(const NumericVector& lhs, const NumericVector& rhs) {
        const int n = lhs.size();
        NumericVector result(n);        
        for (int i = 0; i < n; ++i) {
            result[i] = lhs[i] * rhs[i];
        }
        return result;
    }  
    
    // Syntactic sugar to write: A / B
    template <typename NumericVector>
    NumericVector operator/(const NumericVector& lhs, const NumericVector& rhs) {
        const int n = lhs.size();
        NumericVector result(n);        
        for (int i = 0; i < n; ++i) {
            result[i] = lhs[i] / rhs[i];
        }
        return result;
    }  
    
    // Syntactic sugar to write: const - B
    template <typename NumericVector, typename Real>
    NumericVector operator-(const Real& lhs, const NumericVector& rhs) {
        const int n = rhs.size();
        NumericVector result(n);        
        for (int i = 0; i < n; ++i) {
            result[i] = lhs - rhs[i];
        }
        return result;
    }    
};

namespace ExpressionTemplates {
/*
        Adopted from: http://www.cplusplus.com/forum/general/72582/

*/
    
    struct Scalar; // forward reference
    
	template <class E>
	struct ExpressionTraits {
		// all expressions by const reference
		typedef const E& RefType;
	};

	template <>
	struct ExpressionTraits<Scalar> { // specialization for constants
		// scalar expressions by value
		typedef Scalar RefType;
	};
	
	// basic catch-all expression node
	// L and R must provide operator[](unsigned int) (and work with ExpressionTraits<>)
	// O must provide static function double eval(double,double)
	template <class L, class O, class R>
	struct Expression {
		typedef typename ExpressionTraits<L>::RefType lhsRef;
		typedef typename ExpressionTraits<R>::RefType rhsRef;

		Expression(lhsRef l, rhsRef r) : l(l), r(r) { }

		double operator[](const unsigned int index) const {
			return O::eval(l[index], r[index]);
		}

		lhsRef l;
		rhsRef r;
	};	
	
	struct Scalar {
		Scalar(const double& t) : t(t) { }

		// act like an endless vector of ts
		double operator[](unsigned int) const { return t; }

		const double& t;
	};

	// an operation function object
	struct Plus {
		static double eval(const double a, const double b) { return a + b; }
	};
	
	struct Multiply {
        static double eval(const double a, const double b) { return a * b; }	
	};
	
	struct Minus {
        static double eval(const double a, const double b) { return a - b; }	
	};
	
	struct Divide {
        static double eval(const double a, const double b) { return a / b; }	
	};

    // Synactic sugar starts here	
	template <class L, class R>
	Expression<L,Plus,R> operator+(const L& l, const R& r) {
		return Expression<L,Plus,R>(l, r);
	}
	
	template <class L, class R>
	Expression<L,Multiply,R> operator*(const L& l, const R& r) {
		return Expression<L,Multiply,R>(l, r);
	}
	
	template <class L, class R>
	Expression<L,Minus,R> operator-(const L& l, const R& r) {
		return Expression<L,Minus,R>(l, r);
	}	
	
	template <class L, class R>
	Expression<L,Divide,R> operator/(const L& l, const R& r) {
		return Expression<L,Divide,R>(l, r);
	}	
	
    //scalar - anything
	template <class R>
	Expression<Scalar,Minus,R> operator-(const double& l, const R& r) {
		return Expression<Scalar,Minus,R>(l, r);
	}
};

namespace SimpleLoop {
    
    template <typename NumericVector>
    inline void AlgebraicExpression(NumericVector& result,
            const NumericVector& A, const NumericVector& B,
            const NumericVector& C, const NumericVector& D) {            
        using namespace ComponentWiseOps; // specify operator+
    
        result = A + B + C + D;
        
	    // Compiler generates:
	    // 		tmp1 = A + B;
	    // 		tmp2 = tmp1 + C;
	    // 		tmp3 = tmp2 + D;
	    //      result = tmp3; // copy, often optimized away        
    }

    template <typename NumericVector>
    inline void Transformation(NumericVector& result,
            const NumericVector& A, const NumericVector& B,
            const NumericVector& C, const NumericVector& D) {
            
        const int n = result.size();                      
		for (int i = 0; i < n; ++i) {
		    // Fused all element-wise operations
			result[i] = A[i] + B[i] + C[i] + D[i];
		}		
	}
	
    template <typename NumericVector>
    inline void AlgebraicExpressionTemplate(NumericVector& result,
            const NumericVector& A, const NumericVector& B,
            const NumericVector& C, const NumericVector& D) {            
        using namespace ExpressionTemplates; // Very modern technique
    
        result = A + B + C + D;                
	    // Compiler generates no temporarily intermediates.
		// (A + B + C + D) translates into a tree of operation "types" called expression templates.
	    // Single operator= performs actual transformation by expanding 
	    // and optimizing expression into code.	    
	    // Also provides lazy-evaluation, such that (A + B + C + D)[i] only
	    // evaluates for the i-th entry; particularly useful
	    // for sparse updates
    }	
};

namespace ComplexLoop {

    template <typename NumericVector>
    inline void AlgebraicExpression(
            NumericVector& gradient, NumericVector& hessian,
            const NumericVector& W, const NumericVector& N,
            const NumericVector& D) {            
        using namespace ComponentWiseOps; // specify operator+, etc.
    
        gradient = W * N / D;
        hessian  = W * N / D * (1.0 - N / D); 
                 
    }
    
    template <typename NumericVector>
    inline void Transformation(
            NumericVector& gradient, NumericVector& hessian,
            const NumericVector& W, const NumericVector& N,
            const NumericVector& D) {            
       
        const int n = gradient.size();

    	for (int i = 0; i < n; ++i) {	        
	        gradient[i] = W[i] * N[i] / D[i];
	    }
	    
    	for (int i = 0; i < n; ++i) {	        
	        hessian[i] = W[i] * N[i] / D[i] * (1.0 - N[i] / D[i]);	        
    	}            
    } 
    
    template <typename NumericVector>
    inline void MinOpsTransformation(
            NumericVector& gradient, NumericVector& hessian,
            const NumericVector& W, const NumericVector& N,
            const NumericVector& D) {            
       
        const int n = gradient.size();

    	for (int i = 0; i < n; ++i) {    	    
	        gradient[i] = W[i] * N[i] / D[i];
	    }
	    
    	for (int i = 0; i < n; ++i) {
    	    double ratio = N[i] / D[i];
	        hessian[i] = W[i] * ratio * (1.0 - ratio);
	        // NB: Could mark (N / D) as reusable intermediate in expression template
    	}            
    }   
    
    template <typename NumericVector>
    inline void FusedLoopTransformation(
            NumericVector& gradient, NumericVector& hessian,
            const NumericVector& W, const NumericVector& N,
            const NumericVector& D) {            
       
        const int n = gradient.size();
                
    	for (int i = 0; i < n; ++i) {    	    
	        double ratio = N[i] / D[i];
	        double weightedRatio = W[i] * ratio;
	        gradient[i] = weightedRatio;
	        hessian[i] = weightedRatio * (1.0 - ratio);
	    }	        	          
    }

    template <typename NumericVector>
    inline void TransformationReduction(
            double& totalGradient, double& totalHessian,
            NumericVector& gradient, NumericVector& hessian,
            const NumericVector& W, const NumericVector& N,
            const NumericVector& D) {            
       
        const int n = gradient.size();
                
        totalGradient = 0.0; totalHessian = 0.0;
        double g, h;
    	for (int i = 0; i < n; ++i) {    	    
	        double ratio = N[i] / D[i];
	        double weightedRatio = W[i] * ratio;
	        gradient[i] = g = weightedRatio;
	        hessian[i]  = h = weightedRatio * (1.0 - ratio);
	        totalGradient += g;
	        totalHessian += h;
	    }	        	          
    }
    
    template <typename NumericVector>
    inline void TransformationReduction(
            double& totalGradient, double& totalHessian,           
            const NumericVector& W, const NumericVector& N,
            const NumericVector& D) {            
       
        const int n = W.size();
                
        totalGradient = 0.0; totalHessian = 0.0;        
    	for (int i = 0; i < n; ++i) {    	    
	        double ratio = N[i] / D[i];
	        double weightedRatio = W[i] * ratio;
	        totalGradient += weightedRatio;
	        totalHessian += weightedRatio * (1.0 - ratio);	       
	    }	        	          
    }    
};

namespace Reduction {

    #ifdef USE_OMP
        const static int nThreads = 2;       
    #else
        const static int nThreads = 1;
    #endif

    template <typename NumericVector>
    inline void Serial(double& total, const NumericVector& R) {
        
        const int N = R.size(); 
        double tmpTotal = 0.0; // Compiler should place in register
                       
        int i = 0;
        while (i < N) { // same as for(int i = 0; i < N; ++i), O(N)
            tmpTotal += R[i];
            i += 1;
        }
        total = tmpTotal;
    }
    
    template <typename NumericVector>
    inline void Parallel(double& total, const NumericVector& R) {
                        
        double partialTotal[nThreads];
        const int N = R.size();      
        
        #pragma omp parallel num_threads(nThreads) shared(partialTotal)    
        {   
            double tmpTotal = 0.0; // Private to thread, should be in register or on own cache-line
            #ifdef USE_OMP
                const int thread = omp_get_thread_num();
            #else
                const int thread = 0;
            #endif
            int i = thread; // Get starting index                         
            while (i < N) {  // O(N / nThreads)
                tmpTotal += R[i];
                i += nThreads; // Next interleaved entry
            }           
            partialTotal[thread] = tmpTotal;                                    
        }
                        
        total = 0.0;              
        for (int p = 0; p < nThreads; ++p) { // Can be done in O(log_2 p)        
            total += partialTotal[p];
        }    
    } 
    
    template <typename NumericVector>
    inline void DataLocalParallel(double& total, const NumericVector& R) {
                
        double partialTotal[nThreads];
        const int N = R.size();
        
        const int chunkSize = N / nThreads;
        #pragma omp parallel num_threads(nThreads) shared(partialTotal)    
        {   
            double tmpTotal = 0.0; // Private to thread, should be in register or on own cache-line
            #ifdef USE_OMP
                const int thread = omp_get_thread_num();
            #else
                const int thread = 0;
            #endif
            int i = chunkSize * thread; // Get starting index
            const int end = i + chunkSize;
            while (i < end) { // O( N / nthreads )            
                tmpTotal += R[i];             
                i += 1;
            }          
            partialTotal[thread] = tmpTotal;                                    
        }
                        
        total = 0.0;              
        for (int p = 0; p < nThreads; ++p) { // Can be done in O(log_2 p)
            total += partialTotal[p];
        }    
    }
    
    template <typename NumericVector>
    inline void UnrollParallel(double& total, const NumericVector& R) {
        
        double partialTotal[nThreads];
        const int N = R.size();
        
        const int chunkSize = N / nThreads;
        #pragma omp parallel num_threads(nThreads) shared(partialTotal)    
        {   
            double tmpTotal = 0.0; // Private to thread, should be in register or on own cache-line
            #ifdef USE_OMP
                const int thread = omp_get_thread_num();
            #else
                const int thread = 0;
            #endif
             
            int i = chunkSize * thread; // Get starting index
            const int end = i + chunkSize;
            while (i < end) { // O( N / nthreads )            
                tmpTotal += R[i];             
                i += 1;
                tmpTotal += R[i];
                i += 1; // Currently, chuckSize mod 2 must equal 0                         
            }          
            partialTotal[thread] = tmpTotal;                                    
        }
                        
        total = 0.0;              
        for (int p = 0; p < nThreads; ++p) { // Can be done in O(log_2 p)
            total += partialTotal[p];
        }    
    }     
        
    template <typename NumericVector>
    inline void UsualOMPParallel(double& outTotal, const NumericVector& R) {
        
        const int N = R.size();
        double total = 0.0;
        
        #pragma omp parallel for num_threads(nThreads) reduction(+:total)
        for (int i = 0; i < N; ++i) {
            total += R[i];
        }
        
        outTotal = total;  
    }  
    
    template <typename NumericVector>
    inline void SpecialOMPParallel(double& outTotal, const NumericVector& R) {
        
        const int N = R.size();
        double total = 0.0;
        
        #pragma omp parallel num_threads(nThreads)  
        {   
            double tmpTotal = 0.0; // Private to thread, should be in register or on own cache-line
            #pragma omp for nowait
            for (int i = 0; i < N; ++i) {
                tmpTotal += R[i];
            }
            
            #pragma omp critical
            total += tmpTotal;
        }  
        outTotal = total;
    }     
};

// [[Rcpp::export]]
Rcpp::List execSimpleLoop(
	Rcpp::NumericVector inA, Rcpp::NumericVector inB, 
	Rcpp::NumericVector inC, Rcpp::NumericVector inD,
	int reps, int optLevel) {
	
	int n = inA.size();
	if (n != inB.size() || n != inC.size() || n != inD.size()) {
		throw Rcpp::exception("Unmatched vector lengths");
	}
	
	using namespace Logging; // Can change to Rcpp to ignore logging
	
	NumericVector result(n);
	NumericVector A(inA);
	NumericVector B(inB);
	NumericVector C(inC);
	NumericVector D(inD);
	
	// Reset counters and timing
	clearCounters();		
	struct timeval timeStart, timeEnd;
	gettimeofday(&timeStart,NULL);
	
	for (int r = 0; r < reps; ++r) {
	
	    switch (optLevel) {
	        case 0 : 
	            SimpleLoop::AlgebraicExpression(result, A, B, C, D);
	            break; 
	        case 1 : 
 	            SimpleLoop::Transformation(result, A, B, C, D);
	            break; 
	        case 2 : 
 	            SimpleLoop::AlgebraicExpressionTemplate(result, A, B, C, D);
	            break; 	            	            	            
	        default: 	             	           	               	    
                break;
	    }	    
	}
	
	gettimeofday(&timeEnd,NULL);	
	
	return Rcpp::List::create(
	    Rcpp::Named("result") = result,
	    Rcpp::Named("time") = getTimeDiff(timeStart, timeEnd),
	    Rcpp::Named("ctors") = getCtors() / reps,
	    Rcpp::Named("dtors") = getDtors() / reps,
	    Rcpp::Named("copies") = getCopies() / reps
	    );		
}

// [[Rcpp::export]]
Rcpp::List execComplexLoop(
	Rcpp::NumericVector inW, Rcpp::NumericVector inN, Rcpp::NumericVector inD,
	int reps, int optLevel) {
	
	const int n = inW.size();
	if (n != inN.size() || n != inD.size()) {
		throw Rcpp::exception("Unmatched vector lengths");
	}
	
	using namespace Logging; // Can change to Rcpp to ignore logging
		
	NumericVector gradient(n);
	NumericVector hessian(n);
	NumericVector W(inW);
	NumericVector N(inN);
	NumericVector D(inD);	
		
	// Reset counters and timing
	clearCounters();		
	struct timeval timeStart, timeEnd;
	gettimeofday(&timeStart,NULL);
	
	for (int r = 0; r < reps; ++r) {
	
    	switch (optLevel) { 
    	    case 0 :
    	        ComplexLoop::AlgebraicExpression(gradient, hessian, W, N, D);
    	        break;
    	    case 1 :
    	        ComplexLoop::Transformation(gradient, hessian, W, N, D);
    	        break;
    	    case 2 :
    	        ComplexLoop::MinOpsTransformation(gradient, hessian, W, N, D);
    	        break;
    	    case 3 :
    	        ComplexLoop::FusedLoopTransformation(gradient, hessian, W, N, D);    	        
    	        break;
    	    default:
    	        break;
    	}
	}
	
	gettimeofday(&timeEnd,NULL);	
	
	return Rcpp::List::create(
	    Rcpp::Named("gradient") = gradient,
	    Rcpp::Named("hessian") = hessian,
	    Rcpp::Named("time") = getTimeDiff(timeStart, timeEnd),
	    Rcpp::Named("ctors") = getCtors() / reps,
	    Rcpp::Named("dtors") = getDtors() / reps,
	    Rcpp::Named("copies") = getCopies() / reps	    
	    );		
}

// [[Rcpp::export]]
Rcpp::List execFusedTransformationReduction(
	Rcpp::NumericVector inW, Rcpp::NumericVector inN, Rcpp::NumericVector inD,
	int reps, int optLevel) {
	
	const int n = inW.size();
	if (n != inN.size() || n != inD.size()) {
		throw Rcpp::exception("Unmatched vector lengths");
	}
	
	using namespace Logging; // Can change to Rcpp to ignore logging
		
	NumericVector gradient(n);
	NumericVector hessian(n);
	NumericVector W(inW);
	NumericVector N(inN);
	NumericVector D(inD);
	
	double totalGradient;
	double totalHessian;
		
	// Reset counters and timing
	clearCounters();		
	struct timeval timeStart, timeEnd;
	gettimeofday(&timeStart,NULL);
	
	for (int r = 0; r < reps; ++r) {
	
    	switch (optLevel) { 
    	    case 0 :
    	        ComplexLoop::FusedLoopTransformation(gradient, hessian, W, N, D);
                Reduction::Serial(totalGradient, gradient);
                Reduction::Serial(totalHessian, hessian);
    	        break;
     	    case 1 :
     	        ComplexLoop::TransformationReduction(totalGradient, totalHessian, 
     	            gradient, hessian, W, N, D);
    	        break;
    	    case 2 :
    	        ComplexLoop::TransformationReduction(totalGradient, totalHessian, 
    	            W, N, D);
    	        break;
    	    default:
    	        break;
    	}
	}
	
	gettimeofday(&timeEnd,NULL);	
	
	return Rcpp::List::create(
	    Rcpp::Named("gradient") = totalGradient,
	    Rcpp::Named("hessian") = totalHessian,
	    Rcpp::Named("time") = getTimeDiff(timeStart, timeEnd),
	    Rcpp::Named("ctors") = getCtors() / reps,
	    Rcpp::Named("dtors") = getDtors() / reps,
	    Rcpp::Named("copies") = getCopies() / reps	    
	    );		
}


// [[Rcpp::export]]
Rcpp::List execReduction(Rcpp::NumericVector R,
	int reps, int optLevel) {
	
	const int n = R.size();
	
	double total = 0.0;
		
	// Reset timing	
	struct timeval timeStart, timeEnd;
	gettimeofday(&timeStart,NULL);
	
	for (int r = 0; r < reps; ++r) {
	
    	switch (optLevel) { 
    	    case 0 :
    	        Reduction::Serial(total, R);
    	        break;
    	    case 1 :
    	        Reduction::Parallel(total, R);
    	        break;
    	    case 2 :
    	        Reduction::DataLocalParallel(total, R); 
    	        break;
    	    case 3 :
    	        Reduction::UsualOMPParallel(total, R); 
    	        break;  
    	    case 4 :
    	        Reduction::SpecialOMPParallel(total, R); 
    	        break;    	    	        
    	    case 5 :
    	        Reduction::UnrollParallel(total, R); 
    	        break;    	        
    	    default:
    	        break;
    	}
	}
	
	gettimeofday(&timeEnd,NULL);	
	
	return Rcpp::List::create(
	    Rcpp::Named("total") = total,	   
	    Rcpp::Named("time") = getTimeDiff(timeStart, timeEnd)
	    );		
}
