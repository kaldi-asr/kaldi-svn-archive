// util/kaldi-curlbuf.h

// Copyright 2009-2011  LINSE/UFSC;  Augusto Henrique Hentz

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


/** @file kaldi-curlbuf.h
 *  This is an Kaldi C++ Library header.
 */

#ifndef KALDI_UTIL_KALDI_CURLBUF_H_
#define KALDI_UTIL_KALDI_CURLBUF_H_

#ifdef HAVE_LIBCURL

#include <vector>
#include <string>
#include <streambuf>
#include <curl/curl.h>

namespace kaldi {

class curlbuf : public std::streambuf
{
    public:
        explicit curlbuf(std::size_t put_back = 8, int timeout = 60);
        ~curlbuf();

        bool open(std::string url);
        void close();

        int status_code() const;
        int timeout() const;
        void set_timeout(int timeout);

    private:
        int_type underflow();

        curlbuf(const curlbuf &);
        curlbuf &operator= (const curlbuf &);

        bool fill_buffer();

        static std::size_t header_callback(char *buffer, std::size_t size, std::size_t nitems, void *userp);
        static std::size_t curl_callback(char *buffer, std::size_t size, std::size_t nitems, void *userp);

    private:
        const std::size_t put_back_;
        int timeout_;
        
        std::string url_;
        CURL *curl_;
        CURLM *multi_;
        std::vector<char> buffer_;
        int running_;
        int status_code_;
}; // class curlbuf

}; // namespace kaldi

#endif // HAVE_LIBCURL

#endif // KALDI_UTIL_KALDI_CURLBUF_H_
