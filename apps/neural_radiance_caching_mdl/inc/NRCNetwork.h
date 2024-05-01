#include "shaders/config.h"
#include <cuda.h>
//#include "shaders/neural_radiance_caching.h"

#include <memory>

namespace nrc {

class Network {
public:
	Network();
	~Network(); // = default. Defined in source where Impl is complete.

	template <bool Verbose = false>
	void init(CUstream stream)
	{
		if constexpr (Verbose)
			printConfig_();
		init_(stream);
	}

	template <bool Verbose = false>
	void init(CUstream stream, float lr, float emaDecay = 0.99f)
	{
		if constexpr (Verbose)
			printConfig_();
		init_(stream, lr, emaDecay);
	}

	void destroy();
	
	void setStream(CUstream stream);
	
private:
	// PIMPL idiom to enable #include of this header into pure cpp files
	struct Impl;
	std::unique_ptr<Impl> pImpl{};

	CUstream m_stream{};
	bool m_destroyed{ false };

	void init_(CUstream stream);
	void init_(CUstream stream, float lr, float emaDecay = 0.99f);
	void printConfig_() const;
};

};