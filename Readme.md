Jaxup
============

## Overview

Jaxup is a relatively lightweight and performance-conscious C++ library for parsing and generating [JSON](https://json.org/) documents.  It is built on standard C++11 (with some moderate STL use) and released under the liberal [MIT license](./License.md).  Current build targets include g++ on Ubuntu 18.10, VS 2017 on Windows, and g++ on a Raspberry Pi.  However, it should work on any compiler that supports C++11.

Jaxup supports both a [StAX](https://en.wikipedia.org/wiki/StAX)-style interface and a more user-friendly DOM API.  The underlying StAX interface allows processing of very large documents quickly and with fixed overhead.  Meanwhile, the DOM API simplifies parsing and creation of complex structures.  The two interfaces are intended to work in tandem, allowing you to build/read small DOM nodes inside of a much larger data stream.

## Getting Started

Currently, Jaxup is a header only library, so to get started with it can be a simple as copying the inlude folder somewhere on your include path and including it in files that need to handle JSON.

    #include <jaxup/jaxup.h>

Alternatively, you can build the project and install it with CMake.

    mkdir bin
    cd bin
    cmake ..
    cmake --build .
    cmake --build . --target install

## Unicode support

Currently, Jaxup only handles parsing and generation of UTF-8 documents.  This may be extended in the future, but this covers 99.9% of existing JSON usage.