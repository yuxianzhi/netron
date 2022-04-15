/* jshint esversion: 6 */

var bolt = bolt || {};
var text = text || require('./text');

bolt.Utility = class {
    static NAME_LEN = 128;
    static DataType = ["uint8", "int8", "uint32", "int32", "float16", "int8 + float16",
        "float32", "bnn(0, 1)", "bnn(-1, 1)", "int8 + float32", "uint8_q"];
    static TensorDescBytes = 36;
    static enum(value, type) {
         bolt.Utility._enum = bolt.Utility._enum || new Map([
            [ 'PoolingMode', [ 'POOLING_MAX', 'POOLING_MEAN'] ],
            [ 'RoundMode', [ 'ROUND_CEIL', 'ROUND_FLOOR', 'ROUND_TF_SAME', 'ROUND_TF_VALID', 'ROUND_PREFER_FLOOR', 'ROUND_PREFER_CEIL'] ],
            [ 'ResizeMode', [ 'RESIZE_LINEAR', 'RESIZE_NEAREST', 'RESIZE_CUBIC'] ],
            [ 'CoordinateTransMode', [ 'COORDINATE_TRANS_ALIGN_CORNERS', 'COORDINATE_TRANS_HALF_PIXEL', 'COORDINATE_TRANS_PYTORCH_HALF_PIXEL'
            , 'COORDINATE_TRANS_ASYMMETRIC', 'COORDINATE_TRANS_OUTPUT_HALF_PIXEL'] ],
            [ 'EltwiseMode', [ 'ELTWISE_SUM', 'ELTWISE_MAX', 'ELTWISE_MIN', 'ELTWISE_PROD', 'ELTWISE_SUB', 'ELTWISE_DIV'
            , 'ELTWISE_SQRT', 'ELTWISE_ERF', 'ELTWISE_AND', 'ELTWISE_OR', 'ELTWISE_XOR'] ],
            [ 'ActivationMode', [ 'ACTIVATION_NULL', 'ACTIVATION_RELU', 'ACTIVATION_RELU6', 'ACTIVATION_H_SWISH'
             , 'ACTIVATION_H_SIGMOID', 'ACTIVATION_SIGMOID', 'ACTIVATION_TANH', 'ACTIVATION_GELU', 'ACTIVATION_MISH'
             , 'ACTIVATION_GREATER','ACTIVATION_SOFTPLUS', 'ACTIVATION_EXP', 'ACTIVATION_ABS', 'ACTIVATION_SIGN'
             , 'ACTIVATION_H_SWISH_NODIV', 'ACTIVATION_LOG', 'ACTIVATION_NOT', 'ACTIVATION_NEG', 'ACTIVATION_ROUND'
             , 'ACTIVATION_FLOOR', 'ACTIVATION_CEIL', 'ACTIVATION_SWISH'] ],
            [ 'BilateralSliceApplyMode', [ 'BSLICE_APPLY_NULL', 'BSLICE_APPLY_CONV'] ],
            [ 'ConvolutionMode', [ 'CONVOLUTION_POINTWISE', 'CONVOLUTION_DILATION', 'CONVOLUTION_DEPTHWISE', 'CONVOLUTION_DEPTHWISE_POINTWISE'
             , 'CONVOLUTION_DECONVOLUTION', 'CONVOLUTION_DEPTHWISE_DECONVOLUTION'] ],
            [ 'PadMode', [ 'PAD_CONSTANT', 'PAD_REFLECT', 'PAD_EDGE', 'PAD_SYMMETRIC'] ],
            [ 'CheckMode', [ 'CHECK_EQUAL', 'CHECK_GREATER_EQUAL', 'CHECK_GREAT', 'CHECK_LESS', 'CHECK_LESS_EQUAL', 'CHECK_NOT_EQUAL'] ],
            [ 'ReductionMode', [ 'REDUCTION_SUM', 'REDUCTION_MEAN', 'REDUCTION_STD_DEVIATION', 'REDUCTION_SCALAR_PRODUCT'
             , 'REDUCTION_MAX', 'REDUCTION_MIN', 'REDUCTION_L2'] ],
            [ 'DataConvertType', [ 'F32_to_F32', 'F32_to_F16', 'F32_to_I8'] ],
            [ 'RNNMode', [ 'RNN_RNN', 'RNN_LSTM', 'RNN_GRU', 'RNN_GRU_LBR'] ],
            [ 'ImageFormat', [ 'RGB_SC', 'RGB', 'BGR', 'RGB_RAW', 'RGB_SC_RAW', 'BGR_SC_RAW'] ],
            [ 'ConvolutionPolicy', [ 'CONVOLUTION_NO_TMP_MEM', 'CONVOLUTION_FASTEST', 'CONVOLUTION_TUNNING', 'CONVOLUTION_LIBRARY_SEARCH'] ],
            [ 'ConvolutionForwardAlgorithm', [ 'CONVOLUTION_ALGORITHM_POINTWISE', 'CONVOLUTION_ALGORITHM_DIRECT'
            , 'CONVOLUTION_ALGORITHM_IM2COL_GEMM', 'CONVOLUTION_ALGORITHM_GEMM', 'CONVOLUTION_ALGORITHM_GEMM_ICNCHW'
            , 'CONVOLUTION_ALGORITHM_WINOGRAD', 'CONVOLUTION_ALGORITHM_BNN', 'CONVOLUTION_ALGORITHM_GROUP_DECONV'
            , 'CONVOLUTION_ALGORITHM_GROUP_DECONV', 'CONVOLUTION_ALGORITHM_NULL'] ],
            [ 'DepthwiseConvolutionForwardAlgorithm', [ 'DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT', 'DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT'
            , 'DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT_NO_PADDING', 'DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_3X3S1P1'
            , 'DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM', 'DEPTHWISE_CONVOLUTION_ALGORITHM_NULL'] ]   
        ]);

        if (this._enum.has(type)) {
            const index = parseInt(value, 10);
            const list = this._enum.get(type);
            if (Number.isInteger(index) && index < list.length) {
                return list[index];
            } else {
                alert(index + " is beyond enum " + type + " length.");
                process.exit(-1);
            }
        }
        return value;
    }
};

bolt.ModelFactory = class {
    match(context) {
        return 'bolt.model';
    }
    
    open(context, match) {
        return bolt.Metadata.open(context).then((metadata) => {
            const identifier = context.identifier.toLowerCase();
            const openText = (param, bin) => {
                const reader = new bolt.TextParamReader(param,metadata);
                return new bolt.Model(metadata, reader, bin,identifier);
            };
            return context.request(null, null).then((stream) => {
                const buffer = stream.read();
                return openText(context.stream.peek(), buffer);
            }).catch(() => {
                return openText(context.stream.peek(), null);
            });
        });
    }
};

bolt.Model = class {
    constructor(metadata, param, bin, identifier) {
        this.identifier=identifier;
        this._version = param._version;
        this._description = param._data_type;
        this._graphs = [
            new bolt.Graph(metadata, param, bin)
        ];
    }

    get format() {
        return 'bolt';
    }

    get description() {
        return "precision is " + this._description;
    }

    get version() {
        return this._version;
    }

    get graphs() {
        return this._graphs;
    }
};

bolt.Graph = class {
    constructor(metadata, param, bin) {
        this._inputs = param.inputs;
        this._outputs = param.outputs;
        this._nodes = [];
        const layers = param.layers;
        const args = new Map();
        const arg = (name, type) => {
            if (!args.has(name)) {
                var quantization = null;
                if (param.quantization.has(name)) {
                    quantization = param.quantization.get(name);
                }
                args.set(name, new bolt.Argument(name, type, param.location.get(name), quantization, null));
            }
            return args.get(name);
        };
        for (const layer of layers) {
            let weight = null;
            if (param.weights.has(layer.name)) {
                weight = param.weights.get(layer.name);
            }
            this._nodes.push(new bolt.Node(metadata, layer, weight, arg));
        }
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }
};

bolt.Parameter = class {
    constructor(name, visible, args) {
        this._name = name;
        this._visible = visible;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get arguments() {
        return this._arguments;
    }
};

bolt.Argument = class {
    constructor(name, type, location, quantization, initializer) {
        if (typeof name !== 'string') {
            throw new bolt.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._location = location;
        this._quantization = quantization;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
    }

    get quantization() {
        return this._quantization;
    }

    get location() {
        return this._location;
    }

    get initializer() {
        return this._initializer;
    }
};

var namesMap = new Map();
var inplaceId = 0;

bolt.Node = class {
    constructor(metadata, layer, weight, arg) {
        this._inputs = [];
        this._outputs = [];
        this._chain = [];
        this._name = layer.name || '';
        const type = layer.type;

        this._type = metadata.type(type) || metadata.operator(type) || { name: type };
        const attributeMetadata = this._type && this._type.attributes ? this._type.attributes : [];
        const attributes = layer.attributes;
        const inputs = layer.inputs || [];
        let inputIndex = 0;
        if (this._type && this._type.inputs) {
            for (const inputDef of this._type.inputs) {
                if (inputIndex < inputs.length || inputDef.option != 'optional') {
                    const inputCount = (inputDef.option == 'variadic') ? (inputs.length - inputIndex) : 1;
                    const inputArguments = inputs.slice(inputIndex, inputIndex + inputCount).filter((id) => id != '' || inputDef.option != 'optional').map((id) => arg(id));
                    this._inputs.push(new bolt.Parameter(inputDef.name, true, inputArguments));
                    inputIndex += inputCount;
                }
            }
        }
    
        this._inputs.push(...inputs.slice(inputIndex).map((input, index) => {
            const inputName = ((inputIndex + index) == 0) ? 'input' : (inputIndex + index).toString();
          
            if(namesMap.has(input)){
                input = namesMap.get(input);
            }
            return new bolt.Parameter(inputName, true, [ arg(input) ]);
        }));

        const outputs = layer.outputs || [];
        let outputIndex = 0;
        if (this._type && this._type.outputs) {
            for (const outputDef of this._type.outputs) {
                if (outputIndex < outputs.length || outputDef.option != 'optional') {
                    const outputCount = (outputDef.option == 'variadic') ? (outputs.length - outputIndex) : 1;
                    const outputArguments = outputs.slice(outputIndex, outputIndex + outputCount).map((id) => arg(id));
                    this._outputs.push(new bolt.Parameter(outputDef.name, true, outputArguments));
                    outputIndex += outputCount;
                }
            }
        } 
        this._outputs.push(...outputs.slice(outputIndex).map((output, index) => {
            const outputName = ((outputIndex + index) == 0) ? 'output' : (outputIndex + index).toString();
            if(namesMap.has(output)){
               name = "inplace_" + inplaceId;
               namesMap.set(output, name);
               output = name;
               inplaceId++;
            }else{
                namesMap.set(output, output);
            }
            return new bolt.Parameter(outputName, true, [ arg(output) ]);
        }));
        if (weight != null) {
            if (weight['weight'].length > 0) {
                this._weight('kernel', weight['type'], weight['scale'], weight['weight']);
            }
            if (weight['bias'].length > 0) {
                this._weight('bias', 'float32', null, weight['bias']);
            }
        }

        this._attributes = [];
        let pos = 0;
        let dv = new DataView(attributes.buffer);
        var attributeMap = new Map();
        var arrayMap = new Map();

        for (let i = 0; i < attributeMetadata.length; i++) {
            const metadata = attributeMetadata[i];
            attributeMap.set(metadata.name, i);
            var value = 0;
            switch (metadata.type) {
                case 'int8': {
                    value = dv.getInt8(pos, true);
                    pos += 1;
                    break;
                }
                case 'int8[]': {
                    value = [];
                    for(let i = 0; i < metadata.capacity; i++){
                        value.push(dv.getInt8(pos, true));
                        pos += 1;
                    }
                    if (metadata.length != undefined) {
                        arrayMap.set(metadata.name, metadata.length)
                    }
                    break;
                }
                case 'char[]': {
                    var dataString = "";
                    for(let i = 0; i < metadata.capacity; i++){
                        dataString += String.fromCharCode(dv.getInt8(pos, true));
                        pos += 1;
                    }
                    value = dataString;
                    break;
                }
                case 'bool': {
                    value = Boolean(dv.getInt8(pos, true));
                    pos += 1;
                    break;
                }
                case 'bool[]': {
                    value = [];
                    for(let i = 0; i < metadata.capacity; i++){
                        value.push(dv.getInt8(pos, true));
                        pos += 1;
                    }
                    break;
                }
                case 'int32': {
                    value = dv.getInt32(pos, true);
                    pos += 4;
                    break;
                }
                case 'int32[]': {
                    value = [];
                    for(let i = 0; i < metadata.capacity; i++){
                        value.push(dv.getInt32(pos, true));
                        pos += 4;
                    }
                    if (metadata.length != undefined) {
                        arrayMap.set(metadata.name, metadata.length)
                    }
                    break;
                }
                case 'float32': {
                    value = dv.getFloat32(pos, true);
                    pos += 4;
                    break;
                }
                case 'float32[]': {
                    value = [];
                    for(let i = 0; i < metadata.capacity; i++){
                        value.push(dv.getFloat32(pos, true));
                        pos += 4;
                    }
                    if (metadata.length != undefined) {
                        arrayMap.set(metadata.name, metadata.length)
                    }
                    break;
                }
                case 'TensorDesc': {
                    value = bolt.TensorDesc.read(dv, pos)._dimensions;
                    pos += bolt.Utility.TensorDescBytes;
                    break;
                }
                default: {
                    if (metadata.type) {
                        value = dv.getInt32(pos, true);
                        value = bolt.Utility.enum(value, metadata.type);
                    }
                    break;
                }
            }
            this._attributes.push(new bolt.Attribute(metadata, null, value));
        }

        var lengthSizeArr=[];
        var lengthSize;
        for (var [name, length] of arrayMap) {
            if (attributeMap.has(name) && attributeMap.has(length)) {
                this._attributes[attributeMap.get(name)]._value.length = this._attributes[attributeMap.get(length)].value;
                lengthSize=this._attributes[attributeMap.get(length)];
                if(lengthSizeArr.indexOf(lengthSize)==-1){
                    lengthSizeArr.push(lengthSize);
                }
            }
        }
        for(var i=0;i<lengthSizeArr.length;i++){
            this._attributes.splice(this._attributes.indexOf(lengthSizeArr[i]),1);
        }
        
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get attributes() {
        return this._attributes;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get chain() {
        return this._chain;
    }

    _weight(name, dataType, quantization, data) {
        dataType = dataType || '?';
        const dimensions = [data.length];
        this._inputs.push(new bolt.Parameter(name, true, [
            new bolt.Argument('', null, null, quantization, new bolt.Tensor(new bolt.TensorDesc(dataType, null, dimensions), data))
        ]));
    }

};

bolt.Attribute = class {
    constructor(metadata, key, value) {
        this._type = '';
        this._name = key;
        this._value = value;
        this._visible = true;
        
        if (metadata) {
            this._name = metadata.name;
            if (metadata.type) {
                this._type = metadata.type;
            }

            if (Object.prototype.hasOwnProperty.call(metadata, 'visible') && !metadata.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(metadata, 'default')) {
                if (this._value == metadata.default || (this._value && this._value.toString() == metadata.default.toString())) {
                    this._visible = false;
                }
            }
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible;
    }
};

bolt.Tensor = class {
    constructor(type, data) {
        this._type = type;
        this._data = data;
    }

    get kind() {
        return 'weight';
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state || null;
    }

    get value() {
        const context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        const context = this._context();
        if (context.state) {
            return 'scale   --';
        }
        context.limit = 10000;
        const value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        const context = {};
        context.index = 0;
        context.count = 0;
        context.state = null;

        if (this._type.dataType == '?') {
            context.state = 'Tensor has unknown data type.';
            return context;
        }
        if (!this._type.dimensions) {
            context.state = 'Tensor has no dimensions.';
            return context;
        }

        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.dimensions;
        context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
        context.scale = 0;
        return context;
    }

    _decode(context, dimension) {
        const shape = context.shape.length !== 0 ? context.shape : [ 1 ];
        const results = [];
        const size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (this._type.type) {
                    case 'int8':
                        results.push(context.data.getInt8(context.index, true));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'float16':
                        results.push(context.data.getFloat16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    default:
                            context.state = 'Tensor data type is not implemented.';
                            break;
                }
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
    }
};

bolt.TensorDesc = class {
    constructor(type, format, shape) {
        this._dt = type;
        this._df = format;
        this._dimensions = shape;
    }

    static read(dv, pos) {
        this._dt = dv.getInt32(pos, true);
        pos += 4;

        this._df = dv.getInt32(pos, true);
        pos += 4;

        const nDims = dv.getInt32(pos, true);
        pos += 4;

        this._dimensions = new Array(nDims);
        for(let j = nDims - 1; j >= 0; j--) {
             this._dimensions[j] = dv.getInt32(pos, true);
             pos += 4;
        }
        return this;
    }
    
    get type() {
        return this._dt;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return this._dt + (this._dimensions ? ('[' + this._dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',') + ']') : '');
    }
}

bolt.Metadata = class {
    static open(context) {
        if (bolt.Metadata._metadata) {
            return Promise.resolve(bolt.Metadata._metadata);
        }

        let buffer = context.stream.peek();
        let uint8Array = new Uint8Array(buffer);
        let pos = 0;
        let bufferArr=uint8Array.buffer;
        let dv = new DataView(bufferArr);
        let bolt_version = dv.getInt32(pos, true);

        var json_name = 'bolt-metadata-' + bolt_version + '.json';
        return context.request(json_name, 'utf-8', null).then((data) => {
            bolt.Metadata._metadata = new bolt.Metadata(data);
            return bolt.Metadata._metadata;
        }).catch(() => {
            bolt.Metadata._metadata = new bolt.Metadata(null);
            return bolt.Metadata._metadatas;
        });
    }

    constructor(data) {
        this._operatorMap = new Map();
        this._map = new Map();
        this._attributes = new Map();
        if (data) {
            const items = JSON.parse(data);
            for (const item of items) {
                if (item.name) {
                    this._map.set(item.name, item);
                    if (Object.prototype.hasOwnProperty.call(item, 'operator')) {
                        this._operatorMap.set(item.operator, item.name);
                    }
                }
            }
        }
    }

    operator(code) {
        return this._operatorMap.get(code);
    }

    type(name) {
        return this._map.get(name);
    }

    attribute(type, name) {
        const key = type + ':' + name;
        if (!this._attributes.has(key)) {
            const schema = this.type(type);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    this._attributes.set(type + ':' + attribute.name, attribute);
                }
            }
            if (!this._attributes.has(key)) {
                this._attributes.set(key, null);
            }
        }
        return this._attributes.get(key);
    }

};


bolt.TextParamReader = class {
    constructor(buffer, metadata) {
        let uint8Array = new Uint8Array(buffer);
        let pos = 0;
        let bufferArr=uint8Array.buffer;
        let dv = new DataView(bufferArr);

        this._version = dv.getInt32(pos, true);
        pos += 4;

        // skip magic number
        pos += 4;

        let field = uint8Array.slice(pos, pos + bolt.Utility.NAME_LEN);
        const modelName = text.Reader.open(field, bolt.Utility.NAME_LEN).read();
        pos += bolt.Utility.NAME_LEN;

        this._data_type = bolt.Utility.DataType[dv.getInt32(pos, true)];
        pos += 4;

        const num_inputs = dv.getInt32(pos,true);
        pos += 4;

        const input_names = [];
        for(let i = 0; i < num_inputs; i++) {
            field = uint8Array.slice(pos, pos + bolt.Utility.NAME_LEN);
            input_names.push(text.Reader.open(field, bolt.Utility.NAME_LEN).read());
            pos += bolt.Utility.NAME_LEN;
        }
        this._inputs = [];

        for(let i = 0; i < num_inputs; i++) {
            var tensorRead=bolt.TensorDesc.read(dv,pos);
            var desc =new bolt.TensorDesc(bolt.Utility.DataType[tensorRead._dt],null,tensorRead._dimensions);
            pos += bolt.Utility.TensorDescBytes;
            const tmp = [input_names[i]];
            const input = new bolt.Parameter(input_names[i], true, tmp.map((output) => new bolt.Argument(output, desc, null, null, null)));
            this._inputs.push(input);
        }

        const num_outputs = dv.getInt32(pos,true);
        pos += 4;

        this._outputs = [];
        for(let i = 0; i < num_outputs; i++) {
            field = uint8Array.slice(pos, pos + bolt.Utility.NAME_LEN);
            const name = text.Reader.open(field, bolt.Utility.NAME_LEN).read();
            pos += bolt.Utility.NAME_LEN;

            var desc =new bolt.TensorDesc('float32',null,[]);
            const tmp = [name];
            const output = new bolt.Parameter(name, true, tmp.map((output) => new bolt.Argument(output, desc, null, 0.5, null)));
            this._outputs.push(output);
        }

        const num_operator_specs = dv.getInt32(pos,true);
        pos += 4;

        var layers = [];
        this._location = new Map();
        this._quantization = new Map();
        for (let i = 0; i < num_operator_specs; i++) { 
            const layer = {};
            field = uint8Array.slice(pos,pos + bolt.Utility.NAME_LEN);
            layer.name = text.Reader.open(field, bolt.Utility.NAME_LEN).read();
            pos += bolt.Utility.NAME_LEN;

            const operatorType = dv.getInt32(pos,true);
            layer.type=metadata._operatorMap.get(operatorType);       
            pos += 4;

            const num_inputs = dv.getInt32(pos,true);
            pos += 4;
            layer.inputs = [];
            for (let j = 0; j < num_inputs; j++) {
                field = uint8Array.slice(pos, pos + bolt.Utility.NAME_LEN);
                layer.inputs.push(text.Reader.open(field, bolt.Utility.NAME_LEN).read());
                pos += bolt.Utility.NAME_LEN;
            }

            const num_outputs = dv.getInt32(pos,true);
            pos+=4;
            layer.outputs = [];
            for (let j = 0; j < num_outputs; j++) {
                field = uint8Array.slice(pos, pos + bolt.Utility.NAME_LEN);
                layer.outputs.push(text.Reader.open(field, bolt.Utility.NAME_LEN).read());
                pos += bolt.Utility.NAME_LEN;
            }

            for (let j = 0; j < num_inputs; j++) {
                this._location.set(layer.inputs[j], dv.getInt32(pos,true));
                pos += 4;
            }
            for (let j = 0; j < num_outputs; j++) {
                this._location.set(layer.outputs[j], dv.getInt32(pos,true));
                pos += 4;
            }

            const num_quant_feature = dv.getInt32(pos,true);
            pos += 4;

            if (num_quant_feature > 0) {
                var flag = 0;
                if (num_quant_feature == num_inputs + num_outputs) {  
                    flag = 3;
                } else if (num_quant_feature == num_inputs)  {
                    flag = 1;
                } else if (num_quant_feature == num_outputs)  {
                    flag = 2;
                }
                if (flag == 1 || flag == 3) {
                    for (let j = 0; j < num_inputs; j++) {
                        const scale = this.readScale(dv, pos);
                        this._quantization.set(layer.inputs[j], scale);
                        pos += 4 + scale.length * 4;
                    }
                }
                if (flag == 2 || flag == 3) {
                    for (let j = 0; j < num_outputs; j++) {
                        const scale = this.readScale(dv, pos);
                        this._quantization.set(layer.outputs[j], scale);
                        pos += 4 + scale.length * 4;
                    }
                }
            }

            layer.attributes = uint8Array.slice(pos, pos + metadata.type(layer.type).bytes); 
            if(metadata.type(layer.type).bytes!=null){
                pos += metadata.type(layer.type).bytes;
            } 
            layers.push(layer);
        }
        this._layers = layers;

        const num_weights = dv.getInt32(pos,true);
        pos += 4;

        this._weights = new Map();
        for (let i = 0; i < num_weights; i++) {
            // skip read length
            pos += 4;

            field = uint8Array.slice(pos, pos + bolt.Utility.NAME_LEN);
            const name = text.Reader.open(field, bolt.Utility.NAME_LEN).read();
            pos += bolt.Utility.NAME_LEN;

            const mdt = bolt.Utility.DataType[dv.getInt32(pos, true)];
            pos += 4;

            const weight_bytes = dv.getInt32(pos,true);
            pos += 4;
            let a = pos;
            var weight = null;
            if (mdt == 'float32') {
                const num = weight_bytes / 4;
                weight = new Float32Array(num);
                for (let i = 0; i < num; i++) {
                    weight[i] = dv.getFloat32(pos, true);
                    pos += 4;
                }
            } else if (mdt == 'float16') {
                const num = weight_bytes / 2;   
                weight = new Float32Array(num);
                for (let i = 0; i < num; i++) {
                    weight[i] = dv.getFloat16(pos, true);
                    pos += 2;
                }
            }else if (mdt == 'int32') {
                const num = weight_bytes / 4;
                weight = new Int32Array(num);
                for (let i = 0; i < num; i++) {
                    weight[i] = dv.getInt32(pos, true);
                    pos += 4;
                }
            } else if (mdt == 'int8') {
                const num = weight_bytes;
                weight = new Int8Array(num);
                for (let i = 0; i < num; i++) {
                    weight[i] = dv.getInt8(pos, true);
                    pos += 1;
                }
            }else if (mdt == 'uint8') {
                const num = weight_bytes;
                weight = new Uint8Array(num);
                for (let i = 0; i < num; i++) {
                    weight[i] = dv.getUint8(pos, true);
                    pos += 1;
                }
            } else {
                alert("can not read " + mdt + " type weight.")
            }

            const vec_bytes = dv.getInt32(pos,true);
            pos += 4;
            const vec = new Float32Array(vec_bytes / 4);
            for(let i = 0; i < vec_bytes / 4;i++){
                vec[i] = dv.getFloat32(pos, true); 
                pos += 4;
            }
            
            const num_quant_scale = dv.getInt32(pos,true);
            pos += 4;
            var scale = new Float32Array(num_quant_scale);
            for (var j = 0; j < num_quant_scale; j++) {
                const scale2 = this.readScale(dv, pos);
                pos += 4 + scale2.length * 4;
                if (scale2.length > 0) {
                    scale[j] = scale2[0];
                }
            }
            if (scale.length == 0) {
                scale = null;
            }
            this._weights.set(name, {
                'type': mdt,
                'weight': weight,
                'scale': scale,
                'bias': vec
               });
        }
    }

    readScale(data, pos) {
        const num_scale = data.getInt32(pos, true);
        pos += 4;
        var scale = new Float32Array(num_scale);
        for (let k = 0; k < num_scale; k++) {
            scale[k] = data.getFloat32(pos, true);
            pos += 4;
        }
        return scale;
    }

    get version() {
        return this._version;
    }

    get data_type() {
        return this._data_type;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get layers() {
        return this._layers;
    }

    get weights() {
        return this._weights;
    }

    get location() {
        return this._location;
    }

    get quantization() {
        return this._quantization;
    }
};

bolt.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading bolt model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = bolt.ModelFactory;
}