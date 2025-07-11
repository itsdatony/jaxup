// The MIT License (MIT)
//
// Copyright (c) 2017-2025 Kyle Hawk
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#ifndef JAXUP_NODE_H
#define JAXUP_NODE_H

#include "jaxup_common.h"
#include "jaxup_generator.h"
#include "jaxup_parser.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace jaxup {

enum class JsonNodeType {
	VALUE_OBJECT,
	VALUE_ARRAY,
	VALUE_STRING,
	VALUE_NUMBER_INT,
	VALUE_NUMBER_FLOAT,
	VALUE_BOOLEAN,
	VALUE_NULL
};

static inline std::string getNodeTypeAsString(JsonNodeType t) {
	switch(t) {
	case JsonNodeType::VALUE_OBJECT:
		return "Object";
	case JsonNodeType::VALUE_ARRAY:
		return "Array";
	case JsonNodeType::VALUE_STRING:
		return "String";
	case JsonNodeType::VALUE_NUMBER_INT:
		return "Integer";
	case JsonNodeType::VALUE_NUMBER_FLOAT:
		return "Double";
	case JsonNodeType::VALUE_BOOLEAN:
		return "Boolean";
	case JsonNodeType::VALUE_NULL:
		return "Null";
	default:
		return "Unknown";
	}
}

class JsonNode {
public:
	JsonNode() = default;
	JsonNode(JsonNode&& rhs) {
		type = rhs.type;
		switch (type) {
		case JsonNodeType::VALUE_OBJECT:
			value.object = std::move(rhs.value.object);
			break;
		case JsonNodeType::VALUE_ARRAY:
			value.array = std::move(rhs.value.array);
			break;
		case JsonNodeType::VALUE_STRING:
			value.str = std::move(rhs.value.str);
			break;
		case JsonNodeType::VALUE_NUMBER_INT:
			value.i = rhs.value.i;
			break;
		case JsonNodeType::VALUE_NUMBER_FLOAT:
			value.d = rhs.value.d;
			break;
		case JsonNodeType::VALUE_BOOLEAN:
			value.b = rhs.value.b;
			break;
		default:
			value.i = 0;
		}
		rhs.type = JsonNodeType::VALUE_NULL;
		rhs.value.i = 0;
	}
	~JsonNode() {
		makeNull();
	}

	void copyFrom(const JsonNode& rhs, size_t maxDepth = 50) {
		switch (rhs.type) {
		case JsonNodeType::VALUE_OBJECT:
			if (maxDepth == 0) {
				throw JsonException("Max depth exceeded while copying Object node");
			}
			makeObject();
			value.object->clear();
			value.object->reserve(rhs.size());
			for (const auto& pair : *rhs.value.object) {
				JsonNode newNode;
				newNode.copyFrom(pair.second, maxDepth - 1);
				value.object->emplace_back(pair.first, std::move(newNode));
			}
			break;
		case JsonNodeType::VALUE_ARRAY:
			if (maxDepth == 0) {
				throw JsonException("Max depth exceeded while copying Array node");
			}
			makeArray();
			value.array->clear();
			value.array->reserve(rhs.size());
			for (const auto& node : *rhs.value.array) {
				JsonNode newNode;
				newNode.copyFrom(node, maxDepth - 1);
				value.array->emplace_back(std::move(newNode));
			}
			break;
		case JsonNodeType::VALUE_STRING:
			setString(*rhs.value.str);
			break;
		case JsonNodeType::VALUE_NUMBER_INT:
			setInteger(rhs.value.i);
			break;
		case JsonNodeType::VALUE_NUMBER_FLOAT:
			setDouble(rhs.value.d);
			break;
		case JsonNodeType::VALUE_BOOLEAN:
			setBoolean(rhs.value.b);
			break;
		default:
			makeNull();
		}
	}

	inline void copyTo(JsonNode& rhs, size_t maxDepth = 50) const {
		rhs.copyFrom(*this, maxDepth);
	}

	JsonNodeType getType() const {
		return type;
	}

	inline bool isNumeric() const {
		return this->type == JsonNodeType::VALUE_NUMBER_INT || this->type == JsonNodeType::VALUE_NUMBER_FLOAT;
	}

	int64_t asInteger() const {
		if (this->type == JsonNodeType::VALUE_NUMBER_INT) {
			return this->value.i;
		} else if (this->type == JsonNodeType::VALUE_NUMBER_FLOAT) {
			return static_cast<int64_t>(this->value.d);
		}
		throw JsonException("Attempted to read JSON ", getNodeTypeAsString(this->type), " node as an Integer");
	}

	inline int64_t asInteger(int64_t defaultValue) const {
		if (this->type == JsonNodeType::VALUE_NULL) {
			return defaultValue;
		}
		return asInteger();
	}

	inline int64_t getInteger(const std::string& key) const {
		const auto& node = (*this)[key];
		if (!node.isNumeric()) {
			throw JsonException("Attempted to read field \"", key, "\" as an Integer, but it is of type ", getNodeTypeAsString(node.type));
		}
		return node.asInteger();
	}

	inline int64_t getInteger(const std::string& key, int64_t defaultValue) const {
		const auto& node = (*this)[key];
		if (node.isNull()) {
			return defaultValue;
		} else if (!node.isNumeric()) {
			throw JsonException("Attempted to read field \"", key, "\" as an Integer, but it is of type ", getNodeTypeAsString(node.type));
		}
		return node.asInteger();
	}

	void setInteger(int64_t newValue) {
		setType(JsonNodeType::VALUE_NUMBER_INT);
		this->value.i = newValue;
	}

	inline void operator = (int64_t newValue) {
		setInteger(newValue);
	}

	inline void operator = (int32_t newValue) {
		setInteger(newValue);
	}

	inline void operator = (uint32_t newValue) {
		setInteger(newValue);
	}

	inline void setInteger(const std::string& key, int64_t newValue) {
		(*this)[key].setInteger(newValue);
	}

	double asDouble() const {
		if (this->type == JsonNodeType::VALUE_NUMBER_FLOAT) {
			return this->value.d;
		} else if (this->type == JsonNodeType::VALUE_NUMBER_INT) {
			return static_cast<double>(this->value.i);
		}
		throw JsonException("Attempted to read JSON ", getNodeTypeAsString(this->type), " node as a Double");
	}

	inline double asDouble(double defaultValue) const {
		if (this->type == JsonNodeType::VALUE_NULL) {
			return defaultValue;
		}
		return asDouble();
	}

	inline double getDouble(const std::string& key) const {
		const auto& node = (*this)[key];
		if (!node.isNumeric()) {
			throw JsonException("Attempted to read field \"", key, "\" as a Double, but it is of type ", getNodeTypeAsString(node.type));
		}
		return node.asDouble();
	}

	inline double getDouble(const std::string& key, double defaultValue) const {
		const auto& node = (*this)[key];
		if (node.isNull()) {
			return defaultValue;
		} else if (!node.isNumeric()) {
			throw JsonException("Attempted to read field \"", key, "\" as a Double, but it is of type ", getNodeTypeAsString(node.type));
		}
		return node.asDouble();
	}

	void setDouble(double newValue) {
		setType(JsonNodeType::VALUE_NUMBER_FLOAT);
		this->value.d = newValue;
	}

	inline void operator = (double newValue) {
		setDouble(newValue);
	}

	inline void setDouble(const std::string& key, double newValue) {
		(*this)[key].setDouble(newValue);
	}

	bool asBoolean() const {
		if (this->type == JsonNodeType::VALUE_BOOLEAN) {
			return this->value.b;
		}
		throw JsonException("Attempted to read JSON ", getNodeTypeAsString(this->type), " node as a Boolean");
	}

	inline bool asBoolean(bool defaultValue) const {
		if (this->type == JsonNodeType::VALUE_NULL) {
			return defaultValue;
		}
		return asBoolean();
	}

	inline bool getBoolean(const std::string& key) const {
		const auto& node = (*this)[key];
		if (node.type != JsonNodeType::VALUE_BOOLEAN) {
			throw JsonException("Attempted to read field \"", key, "\" as a Boolean, but it is of type ", getNodeTypeAsString(node.type));
		}
		return node.asBoolean();
	}

	inline bool getBoolean(const std::string& key, bool defaultValue) const {
		const auto& node = (*this)[key];
		if (node.type == JsonNodeType::VALUE_NULL) {
			return defaultValue;
		} else if (node.type != JsonNodeType::VALUE_BOOLEAN) {
			throw JsonException("Attempted to read field \"", key, "\" as a Boolean, but it is of type ", getNodeTypeAsString(node.type));
		}
		return node.asBoolean();
	}

	void setBoolean(bool newValue) {
		setType(JsonNodeType::VALUE_BOOLEAN);
		this->value.b = newValue;
	}

	inline void operator = (bool newValue) {
		setBoolean(newValue);
	}

	inline void setBoolean(const std::string& key, bool newValue) {
		(*this)[key].setBoolean(newValue);
	}

	const std::string& asString() const {
		if (this->type == JsonNodeType::VALUE_STRING) {
			return *this->value.str;
		}
		throw JsonException("Attempted to read JSON ", getNodeTypeAsString(this->type), " node as a String");
	}

	inline const std::string& asString(const std::string& defaultValue) const {
		if (this->type == JsonNodeType::VALUE_NULL) {
			return defaultValue;
		}
		return asString();
	}

	inline const std::string& getString(const std::string& key) const {
		const auto& node = (*this)[key];
		if (node.type != JsonNodeType::VALUE_STRING) {
			throw JsonException("Attempted to read field \"", key, "\" as a String, but it is of type ", getNodeTypeAsString(node.type));
		}
		return node.asString();
	}

	inline const std::string& getString(const std::string& key, const std::string& defaultValue) const {
		const auto& node = (*this)[key];
		if (node.type == JsonNodeType::VALUE_NULL) {
			return defaultValue;
		} else if (node.type != JsonNodeType::VALUE_STRING) {
			throw JsonException("Attempted to read field \"", key, "\" as a String, but it is of type ", getNodeTypeAsString(node.type));
		}
		return node.asString();
	}

	void setString(const std::string& newValue) {
		setType(JsonNodeType::VALUE_STRING);
		new (&this->value.str) StrPtr(new std::string(newValue));
	}

	void setString(const char* newValue) {
		setType(JsonNodeType::VALUE_STRING);
		new (&this->value.str) StrPtr(new std::string(newValue));
	}

	void setString(const char* newValue, size_t size) {
		setType(JsonNodeType::VALUE_STRING);
		new (&this->value.str) StrPtr(new std::string(newValue, size));
	}

	inline void operator = (const std::string& newValue) {
		setString(newValue);
	}

	inline void operator = (const char* newValue) {
		setString(newValue);
	}

	inline void setString(const std::string& key, const std::string& newValue) {
		(*this)[key].setString(newValue);
	}

	bool isNull() const {
		return this->type == JsonNodeType::VALUE_NULL;
	}

	void makeNull() {
		setType(JsonNodeType::VALUE_NULL);
	}

	inline void operator = (std::nullptr_t) {
		makeNull();
	}

	void makeArray() {
		if (this->type == JsonNodeType::VALUE_ARRAY) {
			return;
		}
		setType(JsonNodeType::VALUE_ARRAY);
		new (&this->value.array) ArrayPtr(new std::vector<JsonNode>);
	}

	const JsonNode& operator[](size_t n) const {
		static const JsonNode nullNode;
		if (this->type != JsonNodeType::VALUE_ARRAY || n > this->value.array->size()) {
			return nullNode;
		}
		return this->value.array->at(n);
	}

	JsonNode& operator[](size_t n) {
		if (this->type != JsonNodeType::VALUE_ARRAY) {
			makeArray();
		}
		if (n >= this->value.array->size()) {
			if (n == this->value.array->size()) {
				this->value.array->emplace_back(std::move(JsonNode()));
			} else {
				this->value.array->resize(n + 1);
			}
		}
		return this->value.array->at(n);
	}

	JsonNode& append() {
		if (this->type != JsonNodeType::VALUE_ARRAY) {
			makeArray();
		}
		this->value.array->emplace_back(std::move(JsonNode()));
		return this->value.array->back();
	}

	void makeObject() {
		if (this->type == JsonNodeType::VALUE_OBJECT) {
			return;
		}
		setType(JsonNodeType::VALUE_OBJECT);
		new (&this->value.object) ObjectPtr(new std::vector<std::pair<std::string, JsonNode>>);
	}

	const JsonNode& operator[](const std::string& key) const {
		static const JsonNode nullNode;
		if (this->type != JsonNodeType::VALUE_OBJECT) {
			return nullNode;
		}
		for (auto& pair : *this->value.object) {
			if (pair.first == key) {
				return pair.second;
			}
		}
		return nullNode;
	}

	JsonNode& operator[](const std::string& key) {
		if (this->type != JsonNodeType::VALUE_OBJECT) {
			makeObject();
		}
		for (auto& pair : *this->value.object) {
			if (pair.first == key) {
				return pair.second;
			}
		}
		this->value.object->emplace_back(key, std::move(JsonNode()));
		return this->value.object->back().second;
	}

	JsonNode& append(const std::string& key) {
		if (this->type != JsonNodeType::VALUE_OBJECT) {
			makeObject();
		}
		this->value.object->emplace_back(key, std::move(JsonNode()));
		return this->value.object->back().second;
	}

	const std::pair<const std::string&, const JsonNode&> getField(size_t n) const {
		if (this->type != JsonNodeType::VALUE_OBJECT) {
			throw JsonException("Attempted to get a field out of a JSON ", getNodeTypeAsString(this->type), " node");
		}
		if (n > this->value.object->size()) {
			throw JsonException("Attempted to get a JSON field by index, but the index is out of range");
		}
		auto& val = this->value.object->at(n);
		return {val.first, val.second};
	}

	std::pair<const std::string&, JsonNode&> getField(size_t n) {
		if (this->type != JsonNodeType::VALUE_OBJECT) {
			throw JsonException("Attempted to get a field out of a JSON ", getNodeTypeAsString(this->type), " node");
		}
		if (n > this->value.object->size()) {
			throw JsonException("Attempted to get a JSON field by index, but the index is out of range");
		}
		auto& val = this->value.object->at(n);
		return {val.first, val.second};
	}

	size_t size() const {
		switch (this->type) {
		case JsonNodeType::VALUE_ARRAY:
			return this->value.array->size();
		case JsonNodeType::VALUE_OBJECT:
			return this->value.object->size();
		default:
			return 0;
		}
	}

	template <class dest>
	void write(JsonGenerator<dest>& generator, size_t maxDepth = 50) const {
		switch (type) {
		case JsonNodeType::VALUE_NUMBER_FLOAT:
			generator.write(value.d);
			break;
		case JsonNodeType::VALUE_NUMBER_INT:
			generator.write(value.i);
			break;
		case JsonNodeType::VALUE_NULL:
			generator.write(nullptr);
			break;
		case JsonNodeType::VALUE_BOOLEAN:
			generator.write(value.b);
			break;
		case JsonNodeType::VALUE_STRING:
			generator.write(*value.str);
			break;
		case JsonNodeType::VALUE_ARRAY:
			if (maxDepth == 0) {
				throw JsonException("Max depth exceeded while writing Array node");
			}
			generator.startArray();
			for (const auto& node : *value.array) {
				node.write(generator, maxDepth - 1);
			}
			generator.endArray();
			break;
		case JsonNodeType::VALUE_OBJECT:
			if (maxDepth == 0) {
				throw JsonException("Max depth exceeded while writing Object node");
			}
			generator.startObject();
			for (const auto& pair : *value.object) {
				generator.writeFieldName(pair.first);
				pair.second.write(generator, maxDepth - 1);
			}
			generator.endObject();
			break;
		}
	}

	template <class source>
	void read(JsonParser<source>& parser, size_t maxDepth = 50) {
		JsonToken token = parser.currentToken();
		if (token == JsonToken::NOT_AVAILABLE) {
			// Give a kick start if the stream hasn't been read from
			token = parser.nextToken();
		}

		switch (token) {
		case JsonToken::VALUE_NUMBER_FLOAT:
			setDouble(parser.getDoubleValue());
			break;
		case JsonToken::VALUE_NUMBER_INT:
			setInteger(parser.getIntegerValue());
			break;
		case JsonToken::VALUE_NULL:
			makeNull();
			break;
		case JsonToken::VALUE_TRUE:
			setBoolean(true);
			break;
		case JsonToken::VALUE_FALSE:
			setBoolean(false);
			break;
		case JsonToken::VALUE_STRING:
			setString(parser.getText());
			break;
		case JsonToken::START_ARRAY: {
			if (maxDepth == 0) {
				throw JsonException("Max depth exceeded while parsing Array node");
			}
			makeArray();
			JsonNode newNode;
			JsonToken current = parser.nextToken();
			while (current != JsonToken::END_ARRAY && current != JsonToken::NOT_AVAILABLE) {
				newNode.read(parser, maxDepth - 1);
				this->value.array->emplace_back(std::move(newNode));
				current = parser.currentToken();
			}
		} break;
		case JsonToken::START_OBJECT: {
			if (maxDepth == 0) {
				throw JsonException("Max depth exceeded while parsing Object node");
			}
			makeObject();
			JsonNode newNode;
			std::string fieldName;
			JsonToken current = parser.nextToken();
			while (current == JsonToken::FIELD_NAME) {
				fieldName = parser.getCurrentName();
				current = parser.nextToken();
				newNode.read(parser, maxDepth - 1);
				current = parser.currentToken();
				this->value.object->emplace_back(fieldName, std::move(newNode));
			}
		} break;
		default:
			return;
		}
		parser.nextToken();
	}

private:
	JsonNodeType type = JsonNodeType::VALUE_NULL;
	using StrPtr = std::unique_ptr<std::string>;
	using ArrayPtr = std::unique_ptr<std::vector<JsonNode>>;
	using ObjectPtr = std::unique_ptr<std::vector<std::pair<std::string, JsonNode>>>;
	union Value {
		Value() { i = 0; }
		~Value() {}
		int64_t i;
		double d;
		bool b;
		StrPtr str;
		ArrayPtr array;
		ObjectPtr object;
	} value;
	void setType(JsonNodeType newType) {
		switch (type) {
		case JsonNodeType::VALUE_STRING:
			value.str.~StrPtr();
			break;
		case JsonNodeType::VALUE_ARRAY:
			value.array.~ArrayPtr();
			break;
		case JsonNodeType::VALUE_OBJECT:
			value.object.~ObjectPtr();
			break;
		default:
			break;
		}
		type = newType;
	}
};

template <typename T>
class JsonNodeIterator {
public:
	JsonNodeIterator(T* node, size_t i = 0) : node(node), i(i) {}

	bool operator != (JsonNodeIterator& rhs) const {
		return rhs.node != node || rhs.i != i;
	}

	void operator++() {
		++i;
	}

	std::pair<const std::string&, T&> operator*() const {
		if (node->getType() == JsonNodeType::VALUE_ARRAY) {
			static const std::string dummyString;
			return {dummyString, (*node)[i]};
		} else {
			return node->getField(i);
		}
	}

	std::pair<const std::string&, T&> operator->() const {
		if (node->getType() == JsonNodeType::VALUE_ARRAY) {
			static const std::string dummyString;
			return {dummyString, (*node)[i]};
		} else {
			return node->getField(i);
		}
	}

private:
	T* node;
	size_t i;
};

template<typename T>
inline JsonNodeIterator<T> begin(T& node) {
	return JsonNodeIterator<T>(&node);
}

template<typename T>
inline JsonNodeIterator<T> end(T& node) {
	return JsonNodeIterator<T>(&node, node.size());
}

}

#endif