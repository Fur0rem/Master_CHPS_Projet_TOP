/**
 * @file culkan.h
 * @brief Internal implementation of the Culkan library
 */

#ifndef CULKAN_H
#define CULKAN_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

/**
 * @brief The debug flag. If defined, the library will print errors and other debug information
 */
#define DEBUG 1

/**
 * @brief Enum that holds the different error codes that can be returned by the library
 */
typedef enum {
	NO_ERROR,
	OUT_OF_BOUNDS_BINDING,
	FILE_NOT_FOUND,
	TOO_MANY_INVOCATIONS,
	NOT_ENOUGH_MEMORY,
} CulkanErrCodes;

/**
 * @brief Struct that holds the result of a function that can return an error
 */
typedef struct {
	VkResult vkResult;
	CulkanErrCodes ckResult;
} CulkanResult;

/**
 * @ Macro for a successful CulkanResult
 */
#define CULKAN_SUCCESS                                                                                                                     \
	(CulkanResult) {                                                                                                                   \
		VK_SUCCESS, SUCCESS                                                                                                        \
	}

/**
 * @brief Check for any errors after a Culkan function call
 *
 * It reads the result field of a Culkan instance.
 * If there are errors, it prints the error name and code, line and file where the error occurred, and exits the program.
 * @param culkan the Culkan instance to check the result of
 */
#define culkanCheckError(culkan)                                                                                                           \
	if (DEBUG)                                                                                                                         \
		__checkCulkanResult((culkan)->result, __FILE__, __LINE__);

/**
 * @brief Like culkanCheckError(), but with a custom message
 * @param culkan the Culkan instance to check the result of
 * @param message the message to print if there is an error
 */
#define culkanCheckErrorWithMessage(culkan, message)                                                                                       \
	if (DEBUG) {                                                                                                                       \
		if ((culkan)->result.ckResult != NO_ERROR) {                                                                               \
			fprintf(stderr, "ERROR : %s\n", message);                                                                          \
		}                                                                                                                          \
		__checkCulkanResult((culkan)->result, __FILE__, __LINE__);                                                                 \
	}

/**
 * @brief Like culkanCheckError(), but for vulkan functions
 * @param vkResult the result of the vulkan function
 */
#define vkCheckError(vkResult)                                                                                                             \
	if (DEBUG)                                                                                                                         \
		__checkCulkanResult((CulkanResult){vkResult, NO_ERROR}, __FILE__, __LINE__);

/**
 * @brief Checks for any allocation errors after a malloc call
 *
 * If the allocation failed, it prints where the error occurred and exits the program
 * @param variable the variable to check the allocation of
 */
#define culkanCheckAllocation(variable)                                                                                                    \
	if (DEBUG)                                                                                                                         \
		__checkCulkanResult((variable) == NULL ? (CulkanResult){VK_ERROR_OUT_OF_HOST_MEMORY, NO_ERROR}                             \
						       : (CulkanResult){VK_SUCCESS, NO_ERROR},                                             \
				    __FILE__,                                                                                              \
				    __LINE__);

/**
 * @brief Custom malloc with checks for allocation errors
 * @param type the type of the variable to allocate
 * @param nbElems the number of elements to allocate
 * @return the allocated variable
 */
// Warning that type* should be enclosed in parentheses but if i do it C++ will cry
#define culkanMalloc(type, nbElems)                                                                                                        \
	({                                                                                                                                 \
		type* ptr = (type*)malloc(sizeof(type) * (nbElems));                                                                       \
		culkanCheckAllocation(ptr);                                                                                                \
		ptr;                                                                                                                       \
	})

typedef struct {
	VkBufferCreateInfo* bufferCreateInfoVar;
	VkBuffer* vkBufferVar;

	VkMemoryRequirements memoryRequirementsVar;
	VkMemoryAllocateInfo memoryAllocateInfoVar;

	// TODO : see if they are needed
	VkDevice deviceVar;
	VkDeviceMemory deviceMemoryVar;

	VkDescriptorSetLayoutBinding* layoutBindingVar;
	VkDescriptorPoolSize* poolSizeVar;
	VkDescriptorBufferInfo* bufferInfoVar;

	size_t sizeOfVar;
	void* dataVar;
} GPUVariable;

typedef enum {
	STORAGE_BUFFER = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	UNIFORM_BUFFER = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	OUTPUT_BUFFER  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
} CulkanBindingType;

typedef struct {
	size_t size;
	CulkanBindingType type;
} CulkanBinding;

/*
 * Struct that holds the layout of the shader to communicate with the GPU
 */
typedef struct {
	uint32_t bindingCount;
	CulkanBinding* bindings;
} CulkanLayout;

typedef struct {
	uint16_t x, y, z;
} CulkanInvocations;

typedef struct {
	const CulkanLayout* layout;
	const char* shaderPath;
	CulkanInvocations invocations;
	GPUVariable* variables;
	CulkanResult result;
	VkApplicationInfo appInfo;
	VkInstanceCreateInfo createInfo;
	VkInstance instance;
	uint32_t physicalDeviceCount;
	VkPhysicalDevice* physicalDevices;
	VkPhysicalDevice physicalDevice;
	VkPhysicalDeviceProperties deviceProperties;
	uint32_t queueFamilyCount;
	VkQueueFamilyProperties* queueFamilies;
	VkDeviceQueueCreateInfo queueCreateInfo;
	uint32_t family;
	float* queuePriorities;
	VkDeviceCreateInfo deviceCreateInfo;
	VkDevice device;
	VkPhysicalDeviceMemoryProperties memoryProperties;

	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorPoolSize* poolSizeInfo;
	VkDescriptorPoolSize* pPoolSizes;
	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo;
	VkDescriptorPool descriptorPool;
	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
	VkDescriptorSet descriptorSet;
	VkWriteDescriptorSet** descriptorWritesVar;
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;
	VkPipelineLayout pipelineLayout;

	size_t fileSize;
	char* fileName;
	uint32_t* shaderBuffer;

	VkShaderModuleCreateInfo shaderModuleCreateInfo;
	VkShaderModule shaderModule;
	VkPipelineShaderStageCreateInfo stageCreateInfo;
	VkComputePipelineCreateInfo pipelineCreateInfo;
	VkPipeline pipeline;

	VkCommandPoolCreateInfo commandPoolCreateInfo;
	VkCommandPool commandPool;
	VkCommandBufferAllocateInfo commandBufferAllocateInfo;
	VkCommandBuffer commandBuffer;
	VkCommandBufferBeginInfo commandBufferBeginInfo;

	VkFence computeFence;
	VkFenceCreateInfo fenceCreateInfo;

	VkQueue queue;

	VkCommandBuffer* commandBuffers;
} Culkan;

/**
 * @brief Gets a GPUVariable from a Culkan instance
 * @param culkan the Culkan instance to get the variable from
 * @param binding the binding of the variable to get
 * @return the variable
 */
GPUVariable* culkanGetBinding(Culkan* culkan, uint32_t binding);

/**
 * @brief Writes data to a GPUVariable. Can be preferred over culkanWriteBinding, if you want explicit variable names
 * @param variable the variable to write to
 * @param src the data to write
 * @param result the result of the operation
 */
void culkanWriteGPUVariable(GPUVariable* variable, const void* src, CulkanResult* result);

/**
 * @brief Writes data to a binding of a Culkan instance. Can be preferred over culkanWriteGPUVariable, if you want to write to a binding
 * directly
 * @param culkan the Culkan instance to write to
 * @param binding the binding to write to
 * @param src the data to write
 */
void culkanWriteBinding(Culkan* culkan, uint32_t binding, const void* src);

/**
 * @brief Reads data from a GPUVariable. Can be preferred over culkanReadBinding, if you want explicit variable names
 * @param variable the variable to read from
 * @param dst the destination to write the data to
 * @param result the result of the operation
 */
void culkanReadGPUVariable(GPUVariable* variable, void* dst, CulkanResult* result);

/**
 * @brief Reads data from a binding of a Culkan instance. Can be preferred over culkanReadGPUVariable, if you want to read from a binding
 * directly
 * @param culkan the Culkan instance to read from
 * @param binding the binding to read from
 * @param dst the destination to write the data to
 */
void culkanReadBinding(Culkan* culkan, uint32_t binding, void* dst);

/**
 * @brief Initializes a Culkan instance. It allocates memory for the instance, so it should be freed after use by calling culkanDestroy()
 * @param layout the layout of the shader to use
 * @param shaderPath the path to the shader to use
 * @param workGroups the number of invocations to use
 * @return the created Culkan instance
 */
Culkan* culkanInit(const CulkanLayout* layout, const char* shaderPath, CulkanInvocations invocations);

/**
 * @brief Sets up the Culkan instance.
 * Should be called after writing to the bindings and before running the shader
 * @param culkan the Culkan instance to set up
 */
void culkanSetup(Culkan* culkan);

/**
 * @brief Runs the shader of a Culkan instance. Should be called after setting up the instance.
 * @param culkan the Culkan instance to run
 */
void culkanRun(Culkan* culkan);

/**
 * @brief Frees the memory of a Culkan instance
 * @param culkan the Culkan instance to free the memory of
 */
void culkanDestroy(Culkan* culkan);

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

const char* culkanErrCodeToString(CulkanErrCodes code) {
	switch (code) {
		case NO_ERROR:
			return "No error";
		case OUT_OF_BOUNDS_BINDING:
			return "Out of bounds binding";
		case FILE_NOT_FOUND:
			return "File not found";
		case TOO_MANY_INVOCATIONS:
			return "Too many invocations";
		case NOT_ENOUGH_MEMORY:
			return "Not enough memory";
		default:
			return "Unknown error";
	}
}

/**
 * @brief Check if the result of a Vulkan function is an error
 * @param result The result of the Vulkan function
 * @param file The file where the error happened
 */
static void __checkCulkanResult(CulkanResult result, const char* file, int line) {
	int8_t should_exit = 0;
	if (result.vkResult != VK_SUCCESS) {
		fprintf(stderr, "Vulkan error at %s:%d: %s (%d)\n", file, line, string_VkResult(result.vkResult), result.vkResult);
		should_exit = 1;
	}
	if (result.ckResult != NO_ERROR) {
		fprintf(stderr, "Culkan error at %s:%d: %s\n", file, line, culkanErrCodeToString(result.ckResult));
		should_exit = 1;
	}
	if (should_exit) {
		exit(1);
	}
}

void checkCulkanError(Culkan* culkan, const char* file, int line) {
	__checkCulkanResult(culkan->result, file, line);
}

VkWriteDescriptorSet* createDescriptorSetWrite(VkDescriptorSet descriptorSet, VkDescriptorBufferInfo* bufferInfo, uint32_t binding) {
	VkWriteDescriptorSet* descriptorWrite = culkanMalloc(VkWriteDescriptorSet, 1);
	*descriptorWrite		      = (VkWriteDescriptorSet){
				 .sType		   = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				 .pNext		   = NULL,
				 .dstSet	   = descriptorSet,
				 .dstBinding	   = binding,
				 .dstArrayElement  = 0,
				 .descriptorCount  = 1,
				 .descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				 .pImageInfo	   = NULL,
				 .pBufferInfo	   = bufferInfo,
				 .pTexelBufferView = NULL,
	     };
	return descriptorWrite;
}

VkDescriptorBufferInfo* createDescriptorBufferInfo(VkBuffer buffer, uint32_t size) {
	VkDescriptorBufferInfo* bufferInfo = culkanMalloc(VkDescriptorBufferInfo, 1);
	*bufferInfo			   = (VkDescriptorBufferInfo){
				   .buffer = buffer,
				   .offset = 0,
				   .range  = size,
	       };
	return bufferInfo;
}

VkDescriptorSetLayoutBinding* createDescriptorSetLayoutBinding(uint32_t binding, VkDescriptorType descriptorType,
							       VkShaderStageFlags stageFlags) {
	VkDescriptorSetLayoutBinding* layoutBinding = culkanMalloc(VkDescriptorSetLayoutBinding, 1);
	*layoutBinding				    = (VkDescriptorSetLayoutBinding){
					 .binding	     = binding,
					 .descriptorType     = descriptorType,
					 .descriptorCount    = 1,
					 .stageFlags	     = stageFlags,
					 .pImmutableSamplers = NULL,
	     };
	return layoutBinding;
}

VkBufferCreateInfo* createBufferCreateInfo(uint32_t size, VkBufferUsageFlags usage, uint32_t family) {
	VkBufferCreateInfo* bufferCreateInfo = culkanMalloc(VkBufferCreateInfo, 1);
	*bufferCreateInfo		     = (VkBufferCreateInfo){
			       .sType		      = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			       .pNext		      = NULL,
			       .flags		      = 0,
			       .size		      = size,
			       .usage		      = usage,
			       .sharingMode	      = VK_SHARING_MODE_EXCLUSIVE,
			       .queueFamilyIndexCount = 1,
			       .pQueueFamilyIndices   = &family,
	   };
	return bufferCreateInfo;
}

VkBuffer* createBuffer(VkDevice device, VkBufferCreateInfo* bufferCreateInfo) {
	VkBuffer* buffer = culkanMalloc(VkBuffer, 1);
	VkResult result	 = vkCreateBuffer(device, bufferCreateInfo, NULL, buffer);
	vkCheckError(result);
	return buffer;
}

VkDescriptorPoolSize* createDescriptorPoolSize(VkDescriptorType descriptorType, uint32_t descriptorCount) {
	VkDescriptorPoolSize* poolSize = culkanMalloc(VkDescriptorPoolSize, 1);
	*poolSize		       = (VkDescriptorPoolSize){
				 .type		  = descriptorType,
				 .descriptorCount = descriptorCount,
	     };
	return poolSize;
}

uint32_t* culkanOpenShader(const char* filename, size_t* fileSize, Culkan* culkan) {
	FILE* file = fopen(filename, "rb");
	if (!file) {
		culkan->result.ckResult = FILE_NOT_FOUND;
		culkanCheckErrorWithMessage(culkan, filename);
	}
	fseek(file, 0, SEEK_END);
	*fileSize = ftell(file);
	rewind(file);
	uint32_t* buffer = culkanMalloc(uint32_t, *fileSize);
	size_t n_read	 = fread(buffer, *fileSize, 1, file);
	if (n_read != 1) {
		culkan->result.ckResult = FILE_NOT_FOUND;
		culkanCheckErrorWithMessage(culkan, filename);
	}
	fclose(file);
	culkan->result.ckResult = NO_ERROR;
	return buffer;
}

void freeGPUVariableData(GPUVariable* variable) {
	free(variable->bufferCreateInfoVar);
	free(variable->vkBufferVar);
	free(variable->layoutBindingVar);
	free(variable->poolSizeVar);
	free(variable->bufferInfoVar);
}

void freeGPUVariable(GPUVariable* variable) {
	freeGPUVariableData(variable);
	free(variable);
}

GPUVariable* createGPUVariable(size_t sizeOfVar, VkBufferUsageFlags usage, uint32_t binding, uint32_t family, VkDevice device,
			       VkPhysicalDeviceMemoryProperties* memoryProperties, CulkanResult* result) {
	GPUVariable* variable	      = culkanMalloc(GPUVariable, 1);
	variable->bufferCreateInfoVar = createBufferCreateInfo(sizeOfVar, usage, family);
	variable->vkBufferVar	      = createBuffer(device, variable->bufferCreateInfoVar);
	variable->sizeOfVar	      = sizeOfVar;
	variable->deviceVar	      = device;
	vkGetBufferMemoryRequirements(device, *variable->vkBufferVar, &variable->memoryRequirementsVar);
	variable->memoryAllocateInfoVar = (VkMemoryAllocateInfo){
	    .sType	     = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
	    .pNext	     = NULL,
	    .allocationSize  = variable->memoryRequirementsVar.size,
	    .memoryTypeIndex = 0,
	};
	for (uint32_t i = 0; i < memoryProperties->memoryTypeCount; i++) {
		if (variable->memoryRequirementsVar.memoryTypeBits & (1 << i) &&
		    memoryProperties->memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
			variable->memoryAllocateInfoVar.memoryTypeIndex = i;
			break;
		}
	}
	result->vkResult = vkAllocateMemory(device, &variable->memoryAllocateInfoVar, NULL, &variable->deviceMemoryVar);
	vkCheckError(result->vkResult);
	result->vkResult = vkBindBufferMemory(device, *variable->vkBufferVar, variable->deviceMemoryVar, 0);
	vkCheckError(result->vkResult);

	variable->layoutBindingVar =
	    createDescriptorSetLayoutBinding(binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);
	variable->poolSizeVar	= createDescriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
	variable->bufferInfoVar = createDescriptorBufferInfo(*variable->vkBufferVar, variable->sizeOfVar);

	return variable;
}

GPUVariable* culkanGetBinding(Culkan* culkan, uint32_t binding) {
	if (binding >= culkan->layout->bindingCount) {
		culkan->result = (CulkanResult){VK_ERROR_UNKNOWN, OUT_OF_BOUNDS_BINDING};
		culkanCheckError(culkan);
	}
	return &culkan->variables[binding];
}

void culkanWriteGPUVariable(GPUVariable* variable, const void* src, CulkanResult* result) {
	result->vkResult = vkMapMemory(variable->deviceVar, variable->deviceMemoryVar, 0, VK_WHOLE_SIZE, 0, &variable->dataVar);
	vkCheckError(result->vkResult);
	memcpy(variable->dataVar, src, variable->sizeOfVar);
	vkUnmapMemory(variable->deviceVar, variable->deviceMemoryVar);
}

void culkanWriteBinding(Culkan* culkan, uint32_t binding, const void* src) {
	if (binding >= culkan->layout->bindingCount) {
		culkan->result = (CulkanResult){VK_ERROR_UNKNOWN, OUT_OF_BOUNDS_BINDING};
		return;
	}
	culkanWriteGPUVariable(&culkan->variables[binding], src, &culkan->result);
}

void culkanReadGPUVariable(GPUVariable* variable, void* dst, CulkanResult* result) {
	result->vkResult = vkMapMemory(variable->deviceVar, variable->deviceMemoryVar, 0, VK_WHOLE_SIZE, 0, &variable->dataVar);
	vkCheckError(result->vkResult);
	memcpy(dst, variable->dataVar, variable->sizeOfVar);
	vkUnmapMemory(variable->deviceVar, variable->deviceMemoryVar);
}

void culkanReadBinding(Culkan* culkan, uint32_t binding, void* dst) {
	if (binding >= culkan->layout->bindingCount) {
		culkan->result = (CulkanResult){VK_ERROR_UNKNOWN, OUT_OF_BOUNDS_BINDING};
		culkanCheckError(culkan);
	}
	culkanReadGPUVariable(&culkan->variables[binding], dst, &culkan->result);
}

VkBufferUsageFlags toVkBufferUsageFlags(CulkanBindingType type) {
	switch (type) {
		case STORAGE_BUFFER | OUTPUT_BUFFER:
			return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		case UNIFORM_BUFFER:
			return VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		default:
			return 0;
	}
}

// Allocate all the memory needed for the variables and bindings
void culkanGPUAlloc(Culkan* culkan) {
	culkan->variables = culkanMalloc(GPUVariable, culkan->layout->bindingCount);

	for (uint32_t binding_idx = 0; binding_idx < culkan->layout->bindingCount; binding_idx++) {
		VkBufferUsageFlags usage       = toVkBufferUsageFlags(culkan->layout->bindings[binding_idx].type);
		culkan->variables[binding_idx] = *createGPUVariable(culkan->layout->bindings[binding_idx].size,
								    usage,
								    binding_idx,
								    culkan->family,
								    culkan->device,
								    &culkan->memoryProperties,
								    &culkan->result);
	}
}

void culkanGetPhysicalDevices(Culkan* culkan) {
	culkan->result.vkResult = vkEnumeratePhysicalDevices(culkan->instance, &culkan->physicalDeviceCount, NULL);
	culkanCheckError(culkan);

	culkan->physicalDevices = culkanMalloc(VkPhysicalDevice, culkan->physicalDeviceCount);

	culkan->result.vkResult = vkEnumeratePhysicalDevices(culkan->instance, &culkan->physicalDeviceCount, culkan->physicalDevices);
	culkanCheckError(culkan);

	culkan->physicalDevice = culkan->physicalDevices[0];
}

void culkanCheckForEnoughMemory(Culkan* culkan) {
	uint32_t heap_idx;
	for (heap_idx = 0; heap_idx < culkan->memoryProperties.memoryHeapCount; heap_idx++) {
		int is_heap_big_enough = 1;
		for (uint32_t binding_idx = 0; binding_idx < culkan->layout->bindingCount; binding_idx++) {
			if (culkan->memoryProperties.memoryHeaps[heap_idx].size <
			    culkan->variables[binding_idx].memoryRequirementsVar.size) {
				is_heap_big_enough = 0;
				break;
			}
		}
		if (!is_heap_big_enough) {
			culkan->result.ckResult = NOT_ENOUGH_MEMORY;
			char message[256];
			sprintf(message,
				"Heap %d is not big enough : Requested %lu, available %lu",
				heap_idx,
				culkan->variables[heap_idx].memoryRequirementsVar.size,
				culkan->memoryProperties.memoryHeaps[heap_idx].size);
		}
	}
}

void culkanCheckInvocations(Culkan* culkan) {
	if (culkan->deviceProperties.limits.maxComputeWorkGroupInvocations <
	    culkan->invocations.x * culkan->invocations.y * culkan->invocations.z) {
		culkan->result.ckResult = TOO_MANY_INVOCATIONS;
		char message[70];
		sprintf(message,
			"Max invocations: %d, requested invocations: %d",
			culkan->deviceProperties.limits.maxComputeWorkGroupInvocations,
			culkan->invocations.x * culkan->invocations.y * culkan->invocations.z);
		culkanCheckErrorWithMessage(culkan, message);
	}
}

Culkan* culkanInit(const CulkanLayout* layout, const char* shaderPath, CulkanInvocations invocations) {
	Culkan* culkan	    = culkanMalloc(Culkan, 1);
	culkan->layout	    = layout;
	culkan->shaderPath  = shaderPath;
	culkan->invocations = invocations;

	culkan->appInfo = (VkApplicationInfo){
	    .sType		= VK_STRUCTURE_TYPE_APPLICATION_INFO,
	    .pNext		= NULL,
	    .pApplicationName	= "CulkanApp",
	    .applicationVersion = 0,
	    .pEngineName	= NULL,
	    .engineVersion	= 0,
	    .apiVersion		= VK_API_VERSION_1_3,
	};

	culkan->createInfo = (VkInstanceCreateInfo){
	    .sType		     = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
	    .pNext		     = NULL,
	    .flags		     = 0,
	    .pApplicationInfo	     = &culkan->appInfo,
	    .enabledLayerCount	     = 0,
	    .ppEnabledLayerNames     = NULL,
	    .enabledExtensionCount   = 0,
	    .ppEnabledExtensionNames = NULL,
	};

	culkan->result.vkResult = vkCreateInstance(&culkan->createInfo, NULL, &culkan->instance);
	culkanCheckError(culkan);

	culkanGetPhysicalDevices(culkan);

	vkGetPhysicalDeviceProperties(culkan->physicalDevice, &culkan->deviceProperties);

	culkan->family = 0;

	culkanCheckInvocations(culkan);

	vkGetPhysicalDeviceQueueFamilyProperties(culkan->physicalDevice, &culkan->queueFamilyCount, NULL);
	culkan->queueFamilies = culkanMalloc(VkQueueFamilyProperties, culkan->queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(culkan->physicalDevice, &culkan->queueFamilyCount, culkan->queueFamilies);

	uint32_t family = 0;
	while (family < culkan->queueFamilyCount && !(culkan->queueFamilies[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
		family++;
	}

	if (family == culkan->queueFamilyCount) {
		printf("No compute queue family found\n");
		exit(1);
	}

	culkan->queuePriorities = (float*)(float[]){1.0F}; // Obliged to do double cast because C++ won't allow it otherwise (I hate C++)
	const VkDeviceQueueCreateInfo queueCreateInfo = {
	    .sType	      = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
	    .pNext	      = NULL,
	    .flags	      = 0,
	    .queueFamilyIndex = family,
	    .queueCount	      = 1,
	    .pQueuePriorities = culkan->queuePriorities,
	};

	culkan->deviceCreateInfo = (VkDeviceCreateInfo){
	    .sType		     = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
	    .pNext		     = NULL,
	    .flags		     = 0,
	    .queueCreateInfoCount    = 1,
	    .pQueueCreateInfos	     = &queueCreateInfo,
	    .enabledLayerCount	     = 0,
	    .ppEnabledLayerNames     = NULL,
	    .enabledExtensionCount   = 0,
	    .ppEnabledExtensionNames = NULL,
	    .pEnabledFeatures	     = NULL,
	};

	culkan->result.vkResult = vkCreateDevice(culkan->physicalDevice, &culkan->deviceCreateInfo, NULL, &culkan->device);
	culkanCheckError(culkan);

	vkGetPhysicalDeviceMemoryProperties(culkan->physicalDevice, &culkan->memoryProperties);

	culkanGPUAlloc(culkan);

	culkanCheckError(culkan);

	culkan->descriptorWritesVar = culkanMalloc(VkWriteDescriptorSet*, culkan->layout->bindingCount);
	// Unitialised pointers

	culkanCheckForEnoughMemory(culkan);

	return culkan;
}

void culkanSetup(Culkan* culkan) {

	VkDescriptorSetLayoutBinding* layoutBindings = culkanMalloc(VkDescriptorSetLayoutBinding, culkan->layout->bindingCount);

	for (uint32_t i = 0; i < culkan->layout->bindingCount; i++) {
		layoutBindings[i] = (VkDescriptorSetLayoutBinding){
		    .binding	     = i,
		    .descriptorType  = (VkDescriptorType)culkan->layout->bindings[i].type, // They're the same type, so it's safe to cast.
		    .descriptorCount = 1,
		    .stageFlags	     = VK_SHADER_STAGE_COMPUTE_BIT,
		    .pImmutableSamplers = NULL,
		};
	}

	culkan->descriptorSetLayoutCreateInfo = (VkDescriptorSetLayoutCreateInfo){
	    .sType	  = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	    .pNext	  = NULL,
	    .flags	  = 0,
	    .bindingCount = culkan->layout->bindingCount,
	    .pBindings	  = layoutBindings,
	};

	culkan->result.vkResult =
	    vkCreateDescriptorSetLayout(culkan->device, &culkan->descriptorSetLayoutCreateInfo, NULL, &culkan->descriptorSetLayout);
	culkanCheckError(culkan);

	culkan->poolSizeInfo = createDescriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1);

	culkan->pPoolSizes = culkanMalloc(VkDescriptorPoolSize, culkan->layout->bindingCount);
	for (uint32_t i = 0; i < culkan->layout->bindingCount; i++) {
		culkan->pPoolSizes[i] = (VkDescriptorPoolSize){
		    .type	     = (VkDescriptorType)culkan->layout->bindings[i].type,
		    .descriptorCount = 1,
		};
	}

	culkan->descriptorPoolCreateInfo = (VkDescriptorPoolCreateInfo){
	    .sType	   = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
	    .pNext	   = NULL,
	    .flags	   = 0,
	    .maxSets	   = 1,
	    .poolSizeCount = culkan->layout->bindingCount,
	    .pPoolSizes	   = culkan->pPoolSizes,
	};

	culkan->result.vkResult = vkCreateDescriptorPool(culkan->device, &culkan->descriptorPoolCreateInfo, NULL, &culkan->descriptorPool);
	culkanCheckError(culkan);

	culkan->descriptorSetAllocateInfo = (VkDescriptorSetAllocateInfo){
	    .sType		= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
	    .pNext		= NULL,
	    .descriptorPool	= culkan->descriptorPool,
	    .descriptorSetCount = 1,
	    .pSetLayouts	= &culkan->descriptorSetLayout,
	};

	culkan->result.vkResult = vkAllocateDescriptorSets(culkan->device, &culkan->descriptorSetAllocateInfo, &culkan->descriptorSet);
	culkanCheckError(culkan);

	for (uint32_t i = 0; i < culkan->layout->bindingCount; i++) {
		culkan->descriptorWritesVar[i] = createDescriptorSetWrite(culkan->descriptorSet, culkan->variables[i].bufferInfoVar, i);
		vkUpdateDescriptorSets(culkan->device, 1, culkan->descriptorWritesVar[i], 0, NULL);
	}

	culkan->pipelineLayoutCreateInfo = (VkPipelineLayoutCreateInfo){
	    .sType		    = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
	    .pNext		    = NULL,
	    .flags		    = 0,
	    .setLayoutCount	    = 1,
	    .pSetLayouts	    = &culkan->descriptorSetLayout,
	    .pushConstantRangeCount = 0,
	    .pPushConstantRanges    = NULL,
	};

	culkan->result.vkResult = vkCreatePipelineLayout(culkan->device, &culkan->pipelineLayoutCreateInfo, NULL, &culkan->pipelineLayout);
	culkanCheckError(culkan);

	culkan->shaderBuffer = culkanOpenShader(culkan->shaderPath, &culkan->fileSize, culkan);

	culkan->shaderModuleCreateInfo = (VkShaderModuleCreateInfo){
	    .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
	    .pNext    = NULL,
	    .flags    = 0,
	    .codeSize = culkan->fileSize,
	    .pCode    = culkan->shaderBuffer,
	};

	culkan->result.vkResult = vkCreateShaderModule(culkan->device, &culkan->shaderModuleCreateInfo, NULL, &culkan->shaderModule);
	culkanCheckError(culkan);

	culkan->stageCreateInfo = (VkPipelineShaderStageCreateInfo){
	    .sType		 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
	    .pNext		 = NULL,
	    .flags		 = 0,
	    .stage		 = VK_SHADER_STAGE_COMPUTE_BIT,
	    .module		 = culkan->shaderModule,
	    .pName		 = "main",
	    .pSpecializationInfo = NULL,
	};

	culkan->pipelineCreateInfo = (VkComputePipelineCreateInfo){
	    .sType		= VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
	    .pNext		= NULL,
	    .flags		= 0,
	    .stage		= culkan->stageCreateInfo,
	    .layout		= culkan->pipelineLayout,
	    .basePipelineHandle = VK_NULL_HANDLE,
	    .basePipelineIndex	= 0,
	};

	culkan->result.vkResult =
	    vkCreateComputePipelines(culkan->device, VK_NULL_HANDLE, 1, &culkan->pipelineCreateInfo, NULL, &culkan->pipeline);
	culkanCheckError(culkan);

	culkan->commandPoolCreateInfo = (VkCommandPoolCreateInfo){
	    .sType	      = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
	    .pNext	      = NULL,
	    .flags	      = 0,
	    .queueFamilyIndex = culkan->family,
	};

	culkan->result.vkResult = vkCreateCommandPool(culkan->device, &culkan->commandPoolCreateInfo, NULL, &culkan->commandPool);
	culkanCheckError(culkan);

	culkan->commandBufferAllocateInfo = (VkCommandBufferAllocateInfo){
	    .sType		= VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
	    .pNext		= NULL,
	    .commandPool	= culkan->commandPool,
	    .level		= VK_COMMAND_BUFFER_LEVEL_PRIMARY,
	    .commandBufferCount = 1,
	};

	culkan->result.vkResult = vkAllocateCommandBuffers(culkan->device, &culkan->commandBufferAllocateInfo, &culkan->commandBuffer);
	culkanCheckError(culkan);

	culkan->commandBufferBeginInfo = (VkCommandBufferBeginInfo){
	    .sType	      = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	    .pNext	      = NULL,
	    .flags	      = 0,
	    .pInheritanceInfo = NULL,
	};

	vkBeginCommandBuffer(culkan->commandBuffer, &culkan->commandBufferBeginInfo);
	vkCmdBindPipeline(culkan->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, culkan->pipeline);
	vkCmdBindDescriptorSets(
	    culkan->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, culkan->pipelineLayout, 0, 1, &culkan->descriptorSet, 0, NULL);
	vkCmdDispatch(culkan->commandBuffer, 1, 1, 1);
	vkEndCommandBuffer(culkan->commandBuffer);

	culkan->fenceCreateInfo = (VkFenceCreateInfo){
	    .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
	    .pNext = NULL,
	    .flags = 0,
	};

	culkan->result.vkResult = vkCreateFence(culkan->device, &culkan->fenceCreateInfo, NULL, &culkan->computeFence);
	culkanCheckError(culkan);

	vkGetDeviceQueue(culkan->device, culkan->family, 0, &culkan->queue);
}

void culkanRun(Culkan* culkan) {
	culkan->commandBuffers	  = culkanMalloc(VkCommandBuffer, 1);
	culkan->commandBuffers[0] = culkan->commandBuffer;

	VkSubmitInfo submitInfo = {
	    .sType		  = VK_STRUCTURE_TYPE_SUBMIT_INFO,
	    .pNext		  = NULL,
	    .waitSemaphoreCount	  = 0,
	    .pWaitSemaphores	  = NULL,
	    .pWaitDstStageMask	  = NULL,
	    .commandBufferCount	  = 1,
	    .pCommandBuffers	  = culkan->commandBuffers,
	    .signalSemaphoreCount = 0,
	    .pSignalSemaphores	  = NULL,
	};

	vkQueueSubmit(culkan->queue, 1, &submitInfo, culkan->computeFence);
	vkWaitForFences(culkan->device, 1, &culkan->computeFence, VK_TRUE, UINT64_MAX);
	vkDestroyFence(culkan->device, culkan->computeFence, NULL);
}

void culkanDestroy(Culkan* culkan) {
	free(culkan->shaderBuffer);
	vkDestroyShaderModule(culkan->device, culkan->shaderModule, NULL);
	vkDestroyPipeline(culkan->device, culkan->pipeline, NULL);
	vkDestroyPipelineLayout(culkan->device, culkan->pipelineLayout, NULL);
	vkDestroyDescriptorPool(culkan->device, culkan->descriptorPool, NULL);
	vkDestroyDescriptorSetLayout(culkan->device, culkan->descriptorSetLayout, NULL);
	vkDestroyDevice(culkan->device, NULL);
	vkDestroyInstance(culkan->instance, NULL);
	free(culkan->physicalDevices);
	free(culkan->queueFamilies);

	for (uint32_t i = 0; i < culkan->layout->bindingCount; i++) {
		freeGPUVariableData(&culkan->variables[i]);
	}

	free(culkan->variables);
	free(culkan);
}

#endif