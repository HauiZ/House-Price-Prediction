import { useState, useEffect } from 'react';
import { MapPin, Home, ArrowRight, DollarSign, Check, X, AlertCircle, Info } from 'lucide-react';

export default function App() {
  const [features, setFeatures] = useState({
    address: '',
    area: '',
    frontage: '',
    accessRoad: '',
    houseDirection: '',
    balconyDirection: '',
    floors: '',
    bedrooms: '',
    bathrooms: '',
    legalStatus: '',
    furnitureState: ''
  });

  const [price, setPrice] = useState(null);
  const [priceInBillions, setPriceInBillions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(1);
  const [mapCoordinates, setMapCoordinates] = useState({ lat: 10.762622, lng: 106.660172 }); // Ho Chi Minh City default
  const [error, setError] = useState(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [map, setMap] = useState(null);
  const [marker, setMarker] = useState(null);
  const [districts, setDistricts] = useState([]);
  const [selectedDistrict, setSelectedDistrict] = useState('');

  // API configuration
  const API_URL =  'http://localhost:5000/api';

  // Fetch districts data
  useEffect(() => {
    const fetchDistricts = async () => {
      try {
        const response = await fetch(`${API_URL}/locations`);
        if (response.ok) {
          const data = await response.json();
          setDistricts(data);
        }
      } catch (err) {
        console.error('Failed to fetch districts:', err);
      }
    };

    fetchDistricts();
  }, [API_URL]);

  // Simulate Google Maps functionality since we can't load external API
  useEffect(() => {
    // Mock Google Maps functionality
    const initializeMap = () => {
      const mapElement = document.getElementById('map');
      if (mapElement) {
        // Create a simple map visualization with coordinates display
        setMapLoaded(true);
      }
    };

    setTimeout(initializeMap, 500);
  }, []);

  // Update map when coordinates change
  useEffect(() => {
    if (mapLoaded) {
      // Update map display with new coordinates
      const mapElement = document.getElementById('map');
      if (mapElement) {
        // Get district name based on coordinates (simplified simulation)
        const districtName = selectedDistrict || 'Khu vực chưa xác định';

        mapElement.innerHTML = `
          <div class="flex items-center justify-center h-full w-full bg-blue-50 rounded-lg">
            <div class="text-center">
              <div class="mb-2">
                <div class="w-8 h-8 bg-blue-600 rounded-full mx-auto flex items-center justify-center">
                  <MapPin color="white" size={20} />
                </div>
              </div>
              <p class="font-medium text-blue-800">Vị trí: ${districtName}</p>
              <p class="text-sm text-gray-600">Lat: ${mapCoordinates.lat.toFixed(6)}, Lng: ${mapCoordinates.lng.toFixed(6)}</p>
              <p class="text-xs mt-2 text-gray-500">${features.address || 'Chưa nhập địa chỉ'}</p>
            </div>
          </div>
        `;
      }
    }
  }, [mapCoordinates, mapLoaded, features.address, selectedDistrict]);

  const handleChange = (field, value) => {
    setFeatures(prev => ({ ...prev, [field]: value }));
    setError(null); // Clear any previous errors

    // Reset price prediction when inputs change
    setPrice(null);
    setPriceInBillions(null);
  };

  // Handle district selection
  const handleDistrictChange = (districtName) => {
    setSelectedDistrict(districtName);

    // Find the selected district
    const district = districts.find(d => d.name === districtName);

    if (district) {
      // Simulate different coordinates for different districts
      // In a real app, you'd use actual geocoding
      const baseCoords = { lat: 10.762622, lng: 106.660172 };
      const offset = (district.priceMultiplier - 1) * 0.05;

      setMapCoordinates({
        lat: baseCoords.lat + offset,
        lng: baseCoords.lng + offset
      });
    }
  };

  // Geocode address to get coordinates
  const geocodeAddress = async () => {
    if (!features.address) {
      setError('Vui lòng nhập địa chỉ để định vị trên bản đồ');
      return;
    }

    try {
      setLoading(true);

      // Simulate geocoding delay
      await new Promise(resolve => setTimeout(resolve, 800));

      // In a real application, you would call a geocoding API here
      // For this example, we'll just set random coordinates around HCMC
      const randomLat = 10.762622 + (Math.random() - 0.5) * 0.1;
      const randomLng = 106.660172 + (Math.random() - 0.5) * 0.1;

      setMapCoordinates({
        lat: randomLat,
        lng: randomLng
      });

      // Find the closest district (simplified approach)
      // In a real app, you'd use reverse geocoding
      if (districts.length > 0) {
        const randomIndex = Math.floor(Math.random() * districts.length);
        setSelectedDistrict(districts[randomIndex].name);
      }

      setError(null);
      setLoading(false);
    } catch (error) {
      setError('Lỗi khi định vị địa chỉ.');
      console.error('Geocoding error:', error);
      setLoading(false);
    }
  };

  // Handle prediction API call
  const handlePredict = async () => {
    // Validate inputs
    if (!validateInputs()) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Convert features to appropriate format for API
      const featuresForPrediction = {
        address: features.address,
        area: parseFloat(features.area),
        frontage: parseFloat(features.frontage),
        accessRoad: parseFloat(features.accessRoad),
        houseDirection: features.houseDirection,
        balconyDirection: features.balconyDirection || features.houseDirection, // Use house direction as fallback
        floors: parseInt(features.floors),
        bedrooms: parseInt(features.bedrooms),
        bathrooms: parseInt(features.bathrooms),
        legalStatus: features.legalStatus,
        furnitureState: features.furnitureState,
        coordinates: mapCoordinates
      };

      // Call the prediction API
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(featuresForPrediction),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Server error');
      }

      const data = await response.json();
      setPrice(Math.round(data.prediction));
      setPriceInBillions(data.predictionInBillions);
      setLoading(false);

      // Scroll to results
      setTimeout(() => {
        const resultsElement = document.getElementById('results');
        if (resultsElement) {
          resultsElement.scrollIntoView({ behavior: 'smooth' });
        }
      }, 300);

    } catch (error) {
      console.error("API Error:", error);
      setError('Lỗi kết nối đến máy chủ. Vui lòng thử lại sau.');
      setLoading(false);
    }
  };

  // Validate inputs before proceeding
  const validateInputs = () => {
    // Required fields for step 1
    if (currentStep === 1) {
      if (!features.address) {
        setError('Vui lòng nhập địa chỉ bất động sản');
        return false;
      }
      if (!features.area || isNaN(parseFloat(features.area)) || parseFloat(features.area) <= 0) {
        setError('Vui lòng nhập diện tích hợp lệ');
        return false;
      }
      if (!features.frontage || isNaN(parseFloat(features.frontage)) || parseFloat(features.frontage) <= 0) {
        setError('Vui lòng nhập chiều rộng mặt tiền hợp lệ');
        return false;
      }
      if (!features.accessRoad || isNaN(parseFloat(features.accessRoad)) || parseFloat(features.accessRoad) <= 0) {
        setError('Vui lòng nhập chiều rộng đường vào hợp lệ');
        return false;
      }
      // Also check if location is selected
      if (!selectedDistrict) {
        setError('Vui lòng định vị địa chỉ trên bản đồ');
        return false;
      }
    }

    // Required fields for step 2
    if (currentStep === 2) {
      if (!features.houseDirection) {
        setError('Vui lòng chọn hướng nhà');
        return false;
      }
      if (!features.floors || isNaN(parseInt(features.floors)) || parseInt(features.floors) <= 0) {
        setError('Vui lòng nhập số tầng hợp lệ');
        return false;
      }
      if (!features.bedrooms || isNaN(parseInt(features.bedrooms)) || parseInt(features.bedrooms) < 0) {
        setError('Vui lòng nhập số phòng ngủ hợp lệ');
        return false;
      }
      if (!features.bathrooms || isNaN(parseInt(features.bathrooms)) || parseInt(features.bathrooms) < 0) {
        setError('Vui lòng nhập số phòng tắm hợp lệ');
        return false;
      }
    }

    // Required fields for step 3
    if (currentStep === 3) {
      if (!features.legalStatus) {
        setError('Vui lòng chọn tình trạng pháp lý');
        return false;
      }
      if (!features.furnitureState) {
        setError('Vui lòng chọn tình trạng nội thất');
        return false;
      }
    }

    return true;
  };

  // Move to the next step
  const handleNextStep = () => {
    if (validateInputs()) {
      setCurrentStep(currentStep + 1);
      setError(null);

      // Scroll to top of the form
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  // Move to the previous step
  const handlePrevStep = () => {
    setCurrentStep(currentStep - 1);
    setError(null);

    // Scroll to top of the form
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Reset form
  const handleReset = () => {
    setFeatures({
      address: '',
      area: '',
      frontage: '',
      accessRoad: '',
      houseDirection: '',
      balconyDirection: '',
      floors: '',
      bedrooms: '',
      bathrooms: '',
      legalStatus: '',
      furnitureState: ''
    });
    setPrice(null);
    setPriceInBillions(null);
    setCurrentStep(1);
    setError(null);
    setSelectedDistrict('');
    setMapCoordinates({ lat: 10.762622, lng: 106.660172 });
  };

  // Format price with commas
  const formatPrice = (price) => {
    return new Intl.NumberFormat('vi-VN').format(price);
  };

  // Direction options
  const directionOptions = ["Bắc", "Nam", "Đông", "Tây", "Đông Bắc", "Tây Bắc", "Đông Nam", "Tây Nam"];

  // Legal status options
  const legalStatusOptions = ["Sổ đỏ", "Sổ hồng", "Giấy tờ hợp lệ", "Đang chờ sổ", "Khác"];

  // Furniture state options
  const furnitureStateOptions = ["Không nội thất", "Nội thất cơ bản", "Đầy đủ nội thất", "Cao cấp"];

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        <div className="bg-white rounded-xl shadow-md overflow-hidden">
          {/* Header */}
          <div className="bg-blue-600 p-6">
            <div className="flex items-center">
              <Home className="text-white mr-2" size={28} />
              <h1 className="text-2xl font-bold text-white">Định giá bất động sản</h1>
            </div>
            <p className="text-blue-100 mt-1">Nhập thông tin để nhận định giá chính xác</p>
          </div>

          {/* Progress Steps */}
          <div className="border-b border-gray-200">
            <div className="flex">
              <div
                className={`flex-1 py-4 px-6 text-center ${currentStep >= 1 ? 'text-blue-600 font-medium' : 'text-gray-500'}`}
              >
                <span className={`inline-flex items-center justify-center w-8 h-8 rounded-full mr-2 ${currentStep >= 1 ? 'bg-blue-100 text-blue-600' : 'bg-gray-100'}`}>
                  1
                </span>
                Thông tin cơ bản
              </div>
              <div
                className={`flex-1 py-4 px-6 text-center ${currentStep >= 2 ? 'text-blue-600 font-medium' : 'text-gray-500'}`}
              >
                <span className={`inline-flex items-center justify-center w-8 h-8 rounded-full mr-2 ${currentStep >= 2 ? 'bg-blue-100 text-blue-600' : 'bg-gray-100'}`}>
                  2
                </span>
                Chi tiết thiết kế
              </div>
              <div
                className={`flex-1 py-4 px-6 text-center ${currentStep >= 3 ? 'text-blue-600 font-medium' : 'text-gray-500'}`}
              >
                <span className={`inline-flex items-center justify-center w-8 h-8 rounded-full mr-2 ${currentStep >= 3 ? 'bg-blue-100 text-blue-600' : 'bg-gray-100'}`}>
                  3
                </span>
                Pháp lý & Nội thất
              </div>
            </div>
          </div>

          {/* Form Steps */}
          <div className="p-6">
            {/* Step 1: Basic Info */}
            {currentStep === 1 && (
              <div>
                <h2 className="text-xl font-semibold mb-4">Thông tin cơ bản</h2>

                {/* Address input */}
                <div className="mb-4">
                  <label className="block text-gray-700 text-sm font-medium mb-2">
                    Địa chỉ bất động sản <span className="text-red-500">*</span>
                  </label>
                  <div className="flex">
                    <input
                      type="text"
                      className="flex-1 border border-gray-300 rounded-l-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="Nhập địa chỉ chi tiết"
                      value={features.address}
                      onChange={(e) => handleChange('address', e.target.value)}
                    />
                    <button
                      className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-r-lg transition-colors"
                      onClick={geocodeAddress}
                      disabled={loading}
                    >
                      {loading ? 'Đang xử lý...' : 'Định vị'}
                    </button>
                  </div>
                </div>

                {/* Map view */}
                <div className="mb-4">
                  <div className="flex justify-between items-center mb-2">
                    <label className="block text-gray-700 text-sm font-medium">
                      Khu vực <span className="text-red-500">*</span>
                    </label>
                    <span className="text-sm text-blue-600">
                      {selectedDistrict ? selectedDistrict : 'Chưa xác định'}
                    </span>
                  </div>
                  <div id="map" className="border border-gray-200 rounded-lg h-48 bg-gray-50 overflow-hidden">
                    {/* Map will be rendered here */}
                  </div>
                </div>

                {/* Basic measurements */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div>
                    <label className="block text-gray-700 text-sm font-medium mb-2">
                      Diện tích (m²) <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="number"
                      className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="VD: 100"
                      value={features.area}
                      onChange={(e) => handleChange('area', e.target.value)}
                      min="1"
                    />
                  </div>
                  <div>
                    <label className="block text-gray-700 text-sm font-medium mb-2">
                      Mặt tiền (m) <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="number"
                      className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="VD: 5"
                      value={features.frontage}
                      onChange={(e) => handleChange('frontage', e.target.value)}
                      min="0.1"
                      step="0.1"
                    />
                  </div>
                  <div>
                    <label className="block text-gray-700 text-sm font-medium mb-2">
                      Đường vào (m) <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="number"
                      className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="VD: 8"
                      value={features.accessRoad}
                      onChange={(e) => handleChange('accessRoad', e.target.value)}
                      min="0.1"
                      step="0.1"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Step 2: Design Details */}
            {currentStep === 2 && (
              <div>
                <h2 className="text-xl font-semibold mb-4">Chi tiết thiết kế</h2>

                {/* Direction selection */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                  <div>
                    <label className="block text-gray-700 text-sm font-medium mb-2">
                      Hướng nhà <span className="text-red-500">*</span>
                    </label>
                    <select
                      className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      value={features.houseDirection}
                      onChange={(e) => handleChange('houseDirection', e.target.value)}
                    >
                      <option value="">-- Chọn hướng --</option>
                      {directionOptions.map(direction => (
                        <option key={direction} value={direction}>{direction}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-gray-700 text-sm font-medium mb-2">
                      Hướng ban công
                    </label>
                    <select
                      className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      value={features.balconyDirection}
                      onChange={(e) => handleChange('balconyDirection', e.target.value)}
                    >
                      <option value="">-- Không có ban công --</option>
                      {directionOptions.map(direction => (
                        <option key={direction} value={direction}>{direction}</option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Structure details */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div>
                    <label className="block text-gray-700 text-sm font-medium mb-2">
                      Số tầng <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="number"
                      className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="VD: 3"
                      value={features.floors}
                      onChange={(e) => handleChange('floors', e.target.value)}
                      min="1"
                    />
                  </div>
                  <div>
                    <label className="block text-gray-700 text-sm font-medium mb-2">
                      Số phòng ngủ <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="number"
                      className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="VD: 2"
                      value={features.bedrooms}
                      onChange={(e) => handleChange('bedrooms', e.target.value)}
                      min="0"
                    />
                  </div>
                  <div>
                    <label className="block text-gray-700 text-sm font-medium mb-2">
                      Số phòng tắm <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="number"
                      className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="VD: 2"
                      value={features.bathrooms}
                      onChange={(e) => handleChange('bathrooms', e.target.value)}
                      min="0"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Step 3: Legal & Furniture */}
            {currentStep === 3 && (
              <div>
                <h2 className="text-xl font-semibold mb-4">Pháp lý & Nội thất</h2>

                {/* Legal status */}
                <div className="mb-4">
                  <label className="block text-gray-700 text-sm font-medium mb-2">
                    Tình trạng pháp lý <span className="text-red-500">*</span>
                  </label>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                    {legalStatusOptions.map(status => (
                      <div
                        key={status}
                        className={`cursor-pointer border rounded-lg p-3 transition-colors ${features.legalStatus === status
                            ? 'bg-blue-50 border-blue-500 text-blue-700'
                            : 'border-gray-200 hover:bg-gray-50'
                          }`}
                        onClick={() => handleChange('legalStatus', status)}
                      >
                        <div className="flex items-center">
                          <div className={`w-4 h-4 rounded-full mr-2 border ${features.legalStatus === status
                              ? 'bg-blue-500 border-blue-500'
                              : 'border-gray-400'
                            }`}>
                            {features.legalStatus === status && (
                              <Check size={14} className="text-white" />
                            )}
                          </div>
                          <span>{status}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Furniture state */}
                <div className="mb-4">
                  <label className="block text-gray-700 text-sm font-medium mb-2">
                    Tình trạng nội thất <span className="text-red-500">*</span>
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    {furnitureStateOptions.map(state => (
                      <div
                        key={state}
                        className={`cursor-pointer border rounded-lg p-3 transition-colors ${features.furnitureState === state
                            ? 'bg-blue-50 border-blue-500 text-blue-700'
                            : 'border-gray-200 hover:bg-gray-50'
                          }`}
                        onClick={() => handleChange('furnitureState', state)}
                      >
                        <div className="flex items-center">
                          <div className={`w-4 h-4 rounded-full mr-2 border ${features.furnitureState === state
                              ? 'bg-blue-500 border-blue-500'
                              : 'border-gray-400'
                            }`}>
                            {features.furnitureState === state && (
                              <Check size={14} className="text-white" />
                            )}
                          </div>
                          <span>{state}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Error message */}
            {error && (
              <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-lg flex items-start">
                <AlertCircle className="mr-2 flex-shrink-0 mt-0.5" size={16} />
                <span>{error}</span>
              </div>
            )}

            {/* Navigation buttons */}
            <div className="flex justify-between mt-6">
              {currentStep > 1 ? (
                <button
                  className="bg-gray-100 hover:bg-gray-200 text-gray-800 font-medium py-2 px-6 rounded-lg transition-colors flex items-center"
                  onClick={handlePrevStep}
                >
                  <ArrowRight className="mr-2 rotate-180" size={18} />
                  Quay lại
                </button>
              ) : (
                <div></div>
              )}

              {currentStep < 3 ? (
                <button
                  className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-lg transition-colors flex items-center"
                  onClick={handleNextStep}
                >
                  Tiếp theo
                  <ArrowRight className="ml-2" size={18} />
                </button>
              ) : (
                <button
                  className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-lg transition-colors flex items-center"
                  onClick={handlePredict}
                  disabled={loading}
                >
                  {loading ? (
                    <>Đang tính toán...</>
                  ) : (
                    <>
                      <DollarSign className="mr-2" size={18} />
                      Định giá bất động sản
                    </>
                  )}
                </button>
              )}
            </div>
          </div>

          {/* Results Section */}
          {price !== null && (
            <div id="results" className="border-t border-gray-200 bg-gray-50 p-6">
              <h2 className="text-xl font-semibold mb-4">Kết quả định giá</h2>

              <div className="bg-white border border-gray-200 rounded-lg p-4 mb-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-500 mb-1">Giá ước tính</p>
                    <div className="text-3xl font-bold text-blue-600">{formatPrice(price)} VNĐ</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>)
}